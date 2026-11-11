"""
모의매매 핵심 — 자료 읽기, 신호, 기대값 표, 장부.
과거 재현(backtest.py)과 실시간(live.py)이 이 파일의 같은 함수를 쓴다. 그래서 과거 재현에서 본 규칙과 실제로 도는 규칙이 다를 수 없다.

구성
  1. 메이저 묶음 (BTC·ETH·XRP·SOL·DOGE, 종목당 자금의 8%): 20일 고점 돌파에 사서 10일 최저 종가를 깨면 판다.
  2. 알트 묶음 (최대 12종목, 합계 자금의 55%까지): 신호가 뜨면 '기대값 표'에서 그 상황의 과거 성적을 찾아, 통계적으로 플러스인 칸만 산다.
       신호 종류 6가지 × 시장 상태 4가지 × 과열 여부 2가지 = 48칸
       칸마다 과거의 가상 거래(손절 = 최근 10일 저가 아래, 익절 = 위험의 2배, 최대 28일) 성적을 모은다.
       살 수 있는 칸 = 표본 50건·10주 이상 + 평균이 오차 범위를 빼고도 플러스 + 최근 1년 성적이 마이너스가 아님
       비중 = 손절까지 갔을 때 자금의 0.75%(A등급) 또는 0.4%(B등급)를 잃도록 역산, 종목당 최대 8%
  3. 나머지는 현금.
기대값 표는 매주 그 시점까지 끝난 거래만으로 다시 계산한다 (과거 재현에서도 똑같이 — 미래 성적을 미리 쓰지 않는다).
"""
import concurrent.futures as cf
import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("ml", "ml3", "ml4"):
    sys.path.insert(0, os.path.join(HERE, "..", p))
from feat3 import features3  # noqa: E402
from feat4 import features4  # noqa: E402
from features import features  # noqa: E402

DATA = os.environ.get("LAB_DATA", "lab/.data")
OUT = os.environ.get("PAPER_OUT", "data/paper")
STABLE = set("USDT USDC USDS USDE DAI USD1 PYUSD FDUSD TUSD BUSD RLUSD USDP GUSD EURC USDQ USDG EUR EURI AEUR PAXG XUSD".split())
PEGGED = set("AUD GBP JPY TRY BRL SUSD UST USTC BFUSD USDSB PAX BKRW BVND VAI USDD LUSD USDJ USDN CUSD WBTC WBETH BETH BNSOL STETH WSTETH CBBTC WBNB BTCB".split())
EXCLUDE = STABLE | PEGGED
LEVERAGED = re.compile(r"(UP|DOWN|BULL|BEAR)$")
COST = 0.0025  # 한쪽 기준: 수수료 0.1% + 체결 오차 0.15%
MIN_VALUE, MIN_AGE, MIN_ATR, MIN_NAMES = 1e6, 60, 0.007, 20
MAJORS = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
P = dict(
    start_cash=10000.0, major_weight=0.08, alt_cap=0.08, alt_total=0.55, max_alt=12, risk_a=0.0075, risk_b=0.004, hold_days=28, risk_min=0.04, risk_max=0.25, rr=2.0, gap=5,
    gate_n=50, gate_weeks=10, gate_recent_days=365, gate_recent_n=25, grade_a_lb=0.15, thrust=0.05, thrust_days=3, hot_rsi=70.0, reserve=0.03,
)
TYPES = {
    "BASE_TIGHT": "좁은 횡보 뒤 거래량 3배 돌파", "BASE": "횡보 뒤 거래량 돌파", "STAIR": "계단식 하락 뒤 거래량 돌파", "BRK_CALM": "거래량 동반 20일 고점 돌파",
    "TURN": "하락 구조 전환 뒤 직전 고점 돌파", "PULL50": "주봉 정배열 속 50일선 눌림",
}
STATES = {"HOT": "BTC 급등 중", "THRUST": "알트 동반 돌파", "UP": "BTC 50일선 위", "DOWN": "BTC 50일선 아래"}
STATE_NOTE = {"HOT": "BTC RSI 70 초과", "THRUST": "최근 3일 안에 거래량 동반 60일 고점 돌파 종목이 전체의 5% 이상", "UP": "BTC 종가가 50일선 위", "DOWN": "BTC 종가가 50일선 아래"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------
# 자료
# ----------------------------------------------------------------------------------------
def _coin(sym, rows):
    """[날짜, 시가, 고가, 저가, 종가, 거래대금] 목록 → 배열. 액면 변경으로 보이는 봉 이전은 버린다."""
    if len(rows) < 90:
        return None
    a = np.array([r[1:6] for r in rows], dtype=float)
    dates = [r[0] for r in rows]
    ratio = a[1:, 3] / a[:-1, 3]
    bad = np.where((ratio > 20) | (ratio < 0.05))[0]
    if len(bad):
        a, dates = a[bad[-1] + 1 :], dates[bad[-1] + 1 :]
    if len(dates) < 90 or not np.all(np.isfinite(a[:, :4])) or (a[:, :4] <= 0).any():
        return None
    return dict(sym=sym, dates=dates, o=a[:, 0], h=a[:, 1], l=a[:, 2], c=a[:, 3], v=np.nan_to_num(a[:, 4]))


def load_cache(mkt="binance"):
    """lab 수집기가 받아 둔 전체 기간 일봉 (상장폐지 종목 포함). 오늘의 미완성 봉은 뺀다."""
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    coins = {}
    for path in glob.glob(f"{DATA}/{mkt}/*.json"):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        c = _coin(j["sym"], [r for r in j["rows"] if r[0] < today])
        if c:
            coins[j["sym"]] = c
    return coins


BN = "https://data-api.binance.vision"


def _get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def fetch_live(limit=1000, workers=8):
    """지금 거래 중인 바이낸스 USDT 현물의 최근 일봉. 반환: (완성된 봉만 담은 coins, 오늘 진행 중인 봉 {종목: (시가, 고가, 저가, 현재가)}, 오늘 날짜)"""
    info = _get(f"{BN}/api/v3/exchangeInfo?permissions=SPOT")
    syms = [(s["baseAsset"], s["symbol"]) for s in info["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING" and s["baseAsset"] not in EXCLUDE and not LEVERAGED.search(s["baseAsset"])]
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    def one(bs):
        base, symbol = bs
        try:
            ks = _get(f"{BN}/api/v3/klines?symbol={urllib.parse.quote(symbol)}&interval=1d&limit={limit}")  # 한자 이름 종목도 있다
            return base, [[pd.Timestamp(k[0], unit="ms").strftime("%Y-%m-%d"), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[7])] for k in ks]
        except Exception as e:
            return base, e

    coins, partial, failed = {}, {}, []
    with cf.ThreadPoolExecutor(workers) as ex:
        for base, rows in ex.map(one, syms):
            if isinstance(rows, Exception):
                failed.append(base)
                continue
            if rows and rows[-1][0] == today:
                partial[base] = tuple(rows[-1][1:5])
                rows = rows[:-1]
            c = _coin(base, rows)
            if c:
                coins[base] = c
    return coins, partial, today, failed


# ----------------------------------------------------------------------------------------
# 신호
# ----------------------------------------------------------------------------------------
def build_panel(coins):
    """모든 코인·모든 날의 지표를 계산해 (신호 목록, 날짜별 시장 상태, 날짜 목록)을 돌려준다. coins 에는 장부가 쓸 배열을 덧붙인다."""
    dates = sorted({d for c in coins.values() for d in c["dates"]})
    tix = {d: i for i, d in enumerate(dates)}
    T = len(dates)
    btc = pd.Series(coins["BTC"]["c"], index=coins["BTC"]["dates"])
    parts = []
    btc_rsi = np.full(T, np.nan)
    btc_d50 = np.full(T, np.nan)
    for sym, c in coins.items():
        n = len(c["c"])
        ts = np.array([tix[d] for d in c["dates"]])
        F = features(c["o"], c["h"], c["l"], c["c"], c["v"], btc.reindex(c["dates"]).ffill().to_numpy())
        F3 = features3(c["o"], c["h"], c["l"], c["c"], c["v"], btc.reindex(c["dates"]).ffill().to_numpy())
        F4 = features4(c["o"], c["h"], c["l"], c["c"], c["v"])
        pos = np.full(T, -1, dtype=np.int32)
        pos[ts] = np.arange(n)
        c["pos"], c["last_t"], c["stop_ref"] = pos, int(ts[-1]), F3["stop_ref"].to_numpy(dtype=float)
        if sym in MAJORS:
            c["brk20"] = F["brk20"].to_numpy()
            c["lo10"] = pd.Series(c["c"]).rolling(10).min().shift(1).to_numpy()
        if sym == "BTC":
            btc_rsi[ts], btc_d50[ts] = F["rsi14"].to_numpy(), F["d_sma50"].to_numpy()
        if sym in EXCLUDE or sym == "BTC":
            continue
        d = pd.DataFrame(dict(
            sym=sym, t=ts, bar=np.arange(n), val20=F["val20"].to_numpy(), atr_pct=F["atr_pct"].to_numpy(), roc1=F["roc1"].to_numpy(), rs28=F["rs28"].to_numpy(), max30=F["max30"].to_numpy(),
            dsp=F["days_since_pump"].to_numpy(), brk20=F["brk20"].to_numpy(), brk60=F["brk60"].to_numpy(), d_sma50=F["d_sma50"].to_numpy(), vr=F3["vr"].to_numpy(), range60=F3["range60_prev"].to_numpy(),
            w_align=F3["w_align"].to_numpy(), touch50=F3["touch50_5"].to_numpy(), rev=F3["rev_bar"].to_numpy(), turn10=F3["turn10"].to_numpy(), brk_piv=F3["brk_piv"].to_numpy(), drop30=F4["drop30p"].to_numpy(),
        ))
        parts.append(d[(d.bar >= MIN_AGE) & (d.val20 >= MIN_VALUE) & (d.atr_pct >= MIN_ATR)])
    df = pd.concat(parts, ignore_index=True)
    df = df[df.groupby("t").sym.transform("count") >= MIN_NAMES].reset_index(drop=True)
    # 날짜별 시장 상태
    brk = ((df.brk60 > 0.5) & (df.vr >= 2)).groupby(df.t).mean().reindex(range(T)).fillna(0.0)
    thrust = brk.rolling(P["thrust_days"], min_periods=1).max() >= P["thrust"]
    state = np.where(btc_rsi > P["hot_rsi"], "HOT", np.where(thrust.to_numpy(), "THRUST", np.where(btc_d50 > 0, "UP", "DOWN")))
    mkt = pd.DataFrame(dict(date=dates, state=state, btc_rsi=btc_rsi, btc_d50=btc_d50, mkt_brk=brk.to_numpy(), names=df.groupby("t").sym.count().reindex(range(T)).fillna(0).to_numpy()))
    # 신호 종류 (위에서부터 먼저 해당하는 것 하나)
    g = df
    conds = [
        (g.range60 < 0.45) & (g.brk60 == 1) & (g.vr >= 3) & (g.dsp > 120),
        (g.range60 < 0.60) & (g.brk60 == 1) & (g.vr >= 2),
        (g.drop30 <= -0.25) & ((g.brk20 == 1) | (g.brk_piv == 1)) & (g.vr >= 2),
        (g.brk20 == 1) & (g.vr >= 2) & (g.atr_pct <= 0.10),
        (g.turn10 == 1) & (g.brk_piv == 1) & (g.vr >= 1.5),
        (g.w_align == 1) & (g.touch50 == 1) & (g.rev == 1) & (g.d_sma50 > -0.03),
    ]
    typ = np.select([c.to_numpy() for c in conds], list(TYPES), default="")
    sig = df[typ != ""].copy()
    sig["type"] = typ[typ != ""]
    sig = sig[~sig.sym.isin(MAJORS)]  # 메이저는 메이저 묶음에서만 다룬다
    # 과열: 당일 +15% 이상이거나, 28일간 BTC보다 20%p 넘게 앞섰거나, 30일 안에 하루 +29% 넘게 오른 날이 있었음
    sig["flagged"] = ((sig.roc1 >= 0.15) | (sig.rs28 >= 0.20) | (sig.max30 >= 0.29)).astype(int)
    # 같은 코인·같은 종류 신호는 5일 안에 다시 세지 않는다
    sig = sig.sort_values(["sym", "type", "bar"])
    keep = np.ones(len(sig), bool)
    last = (None, None, -(10**9))
    for k, (s, ty, b) in enumerate(zip(sig.sym.to_numpy(), sig.type.to_numpy(), sig.bar.to_numpy())):
        if (s, ty) == last[:2] and b - last[2] <= P["gap"]:
            keep[k] = False
        else:
            last = (s, ty, b)
    sig = sig[keep]
    sig["state"] = state[sig.t.to_numpy()]
    sig["date"] = np.array(dates)[sig.t.to_numpy()]
    sig = sig[["sym", "t", "bar", "date", "type", "state", "flagged", "range60", "vr", "roc1", "val20"]].sort_values(["t", "sym"]).reset_index(drop=True)
    return sig, mkt, dates


def clamp_risk(entry, stop_ref):
    r = (entry - stop_ref * 0.99) / entry if (stop_ref is not None and np.isfinite(stop_ref)) else 0.08
    return float(min(max(r, P["risk_min"]), P["risk_max"]))


def shadow(sig, coins):
    """신호마다 가상 거래 결과 (자금과 무관하게 전부). 기대값 표의 재료다."""
    out = []
    for sym, bar in zip(sig.sym.to_numpy(), sig.bar.to_numpy()):
        c = coins[sym]
        o, h, l, cl, dates = c["o"], c["h"], c["l"], c["c"], c["dates"]
        n, i = len(cl), int(bar)
        if i + 1 >= n:
            out.append((None, None, np.nan, np.nan, np.nan, False))
            continue
        entry = o[i + 1]
        risk = clamp_risk(entry, c["stop_ref"][i])
        stop, tgt = entry * (1 - risk), entry * (1 + P["rr"] * risk)
        px, dx = None, None
        end = min(i + P["hold_days"], n - 1)
        for d in range(i + 1, end + 1):
            if l[d] <= stop:
                px, dx = min(o[d], stop), d
                break
            if h[d] >= tgt:
                px, dx = max(o[d], tgt), d
                break
        if px is None and i + P["hold_days"] <= n - 1:
            px, dx = cl[end], end
        if px is None:
            out.append((dates[i + 1], None, np.nan, np.nan, risk, False))
            continue
        ret = px / entry * (1 - COST) / (1 + COST) - 1
        out.append((dates[i + 1], dates[dx], ret, ret / risk, risk, True))
    res = pd.DataFrame(out, columns=["entry_date", "exit_date", "ret", "R", "risk", "done"], index=sig.index)
    return pd.concat([sig[["sym", "date", "type", "state", "flagged"]], res], axis=1)


def compute_gates(sh, asof):
    """asof 이전에 끝난 가상 거래만으로 칸별 성적과 매수 가능 여부를 계산한다."""
    e = sh[sh.done.astype(bool) & (sh.exit_date < asof)]
    if not len(e):
        return {}
    e = e.assign(week=pd.to_datetime(e.entry_date).dt.strftime("%G-%V"))
    cut = (pd.Timestamp(asof) - pd.Timedelta(days=P["gate_recent_days"])).strftime("%Y-%m-%d")

    def stat(g):
        wk = g.groupby("week").R.mean()
        n, weeks, mean = len(g), len(wk), float(g.R.mean())
        se = float(wk.std(ddof=1) / np.sqrt(weeks)) if weeks >= 2 else float("inf")
        rec = g[g.entry_date >= cut]
        enough = n >= P["gate_n"] and weeks >= P["gate_weeks"]
        lb = mean - se
        ok = bool(enough and lb > 0 and (len(rec) < P["gate_recent_n"] or rec.R.mean() > 0))
        return dict(n=int(n), weeks=int(weeks), meanR=mean, se=se if np.isfinite(se) else None, lb=lb if np.isfinite(lb) else None, meanRet=float(g.ret.mean()), win=float((g.ret > 0).mean()),
                    recentN=int(len(rec)), recentR=float(rec.R.mean()) if len(rec) else None, enough=bool(enough), ok=ok, grade=("A" if lb >= P["grade_a_lb"] else "B") if ok else None)

    cells = {f"{k[0]}|{k[1]}|{k[2]}": stat(g) for k, g in e.groupby(["type", "state", "flagged"])}
    parents = {f"{k[0]}|{k[1]}": stat(g) for k, g in e.groupby(["type", "state"])}
    gates = {}
    for ty in TYPES:
        for st in STATES:
            for fl in (0, 1):
                c = cells.get(f"{ty}|{st}|{fl}")
                if c and c["enough"]:
                    gates[f"{ty}|{st}|{fl}"] = dict(c, level="칸")
                elif fl == 0 and parents.get(f"{ty}|{st}", {}).get("enough"):
                    # 과열 아닌 칸의 표본이 모자라면 (종류 × 시장 상태) 전체 성적을 빌려 쓴다. 과열 칸은 자기 표본이 있어야만 산다.
                    gates[f"{ty}|{st}|{fl}"] = dict(parents[f"{ty}|{st}"], level="종류×상태", own=c)
                else:
                    gates[f"{ty}|{st}|{fl}"] = dict(c or dict(n=0, weeks=0), ok=False, grade=None, level="표본 부족")
    return gates


# ----------------------------------------------------------------------------------------
# 장부 — 하루를 세 단계로 나눈다: 시가(대기 주문 체결) → 장중(손절·익절) → 종가(기간 만료·평가·새 주문)
# ----------------------------------------------------------------------------------------
def new_state(cash, date):
    return dict(startedAt=date, initialCash=cash, cash=cash, positions=[], majors={}, pending=[], openDone=None, closeDone=None, equity=[], trades=[], seq=0, lastEquity=cash, peak=cash, decisions=[], market=None)


def _close_trade(st, p, date, px, reason):
    proceeds = p["qty"] * px * (1 - COST)
    st["cash"] += proceeds
    ret = proceeds / p["value"] - 1
    st["trades"].append(dict(id=p["id"], sleeve=p["sleeve"], sym=p["sym"], type=p.get("type"), state=p.get("state"), flagged=p.get("flagged"), grade=p.get("grade"), entryDate=p["entryDate"], exitDate=date,
                             entry=p["entry"], exit=float(px), ret=float(ret), R=float(ret / p["risk"]) if p.get("risk") else None, pnl=float(proceeds - p["value"]), reason=reason,
                             days=int((pd.Timestamp(date) - pd.Timestamp(p["entryDate"])).days) + 1))


def step_open(st, date, px):
    eq = st["lastEquity"]
    for od in st["pending"]:
        b = px(od["sym"])
        if b is None:
            continue
        o = float(b[0])
        st["seq"] += 1
        if od["kind"] == "alt_buy":
            risk = clamp_risk(o, od.get("stopRef"))
            value = min(eq * P["alt_cap"], eq * od["riskFrac"] / risk, st["cash"] - eq * P["reserve"])
            if value < eq * 0.004:
                continue
            st["cash"] -= value
            st["positions"].append(dict(id=st["seq"], sleeve="alt", sym=od["sym"], type=od["type"], state=od["state"], flagged=od["flagged"], grade=od["grade"], cell=od.get("cell"), signalDate=od["signalDate"], entryDate=date,
                                        entry=o, qty=value / (o * (1 + COST)), value=float(value), risk=risk, stop=o * (1 - risk), target=o * (1 + P["rr"] * risk), last=o))
        elif od["kind"] == "major_buy" and od["sym"] not in st["majors"]:
            value = min(eq * P["major_weight"], st["cash"] - eq * 0.01)
            if value < eq * 0.01:
                continue
            st["cash"] -= value
            st["majors"][od["sym"]] = dict(id=st["seq"], sleeve="major", sym=od["sym"], type="MAJOR", entryDate=date, signalDate=od["signalDate"], entry=o, qty=value / (o * (1 + COST)), value=float(value), last=o)
        elif od["kind"] == "major_sell" and od["sym"] in st["majors"]:
            _close_trade(st, st["majors"].pop(od["sym"]), date, o, "10일 최저 종가 이탈")
    st["pending"] = []
    st["openDone"] = date


def step_intraday(st, date, px):
    keep = []
    for p in st["positions"]:
        b = px(p["sym"])
        if b is None:
            keep.append(p)
            continue
        o, h, l, c = b
        p["last"] = float(c)
        if l <= p["stop"]:
            _close_trade(st, p, date, min(o, p["stop"]), "손절")
        elif h >= p["target"]:
            _close_trade(st, p, date, max(o, p["target"]), "익절 (위험의 2배)")
        else:
            keep.append(p)
    st["positions"] = keep


def step_close(st, date, px, ended, sigs, mkt, gates, major_sig, stop_ref):
    """ended(sym) = 그 코인의 자료가 끝났는가(상장폐지), sigs = 오늘 신호 목록, major_sig(sym) = 'enter'|'exit'|None, stop_ref(sym) = 최근 10일 저가"""
    keep = []
    for p in st["positions"]:
        b = px(p["sym"])
        if b is None:
            if ended(p["sym"]):
                _close_trade(st, p, date, p["last"], "거래 종료")
            else:
                keep.append(p)
            continue
        p["last"] = float(b[3])
        if (pd.Timestamp(date) - pd.Timestamp(p["entryDate"])).days >= P["hold_days"] - 1:
            _close_trade(st, p, date, b[3], "28일 만료")
        else:
            keep.append(p)
    st["positions"] = keep
    for sym in list(st["majors"]):
        b = px(sym)
        if b is not None:
            st["majors"][sym]["last"] = float(b[3])
        elif ended(sym):
            m = st["majors"].pop(sym)
            _close_trade(st, m, date, m["last"], "거래 종료")
    alt_val = sum(p["qty"] * p["last"] for p in st["positions"])
    maj_val = sum(p["qty"] * p["last"] for p in st["majors"].values())
    eq = st["cash"] + alt_val + maj_val
    st["lastEquity"], st["peak"] = eq, max(st.get("peak", eq), eq)
    st["equity"].append([date, round(eq, 2), round(st["cash"], 2), len(st["positions"]), len(st["majors"]), round((alt_val + maj_val) / eq, 4) if eq > 0 else 0])
    # 메이저 주문
    cash_avail = st["cash"]
    for sym in MAJORS:
        s = major_sig(sym)
        if s == "enter" and sym not in st["majors"]:
            st["pending"].append(dict(kind="major_buy", sym=sym, signalDate=date))
            cash_avail -= eq * P["major_weight"]
        elif s == "exit" and sym in st["majors"]:
            st["pending"].append(dict(kind="major_sell", sym=sym, signalDate=date))
    # 알트 후보 → 기대값 표 → 주문
    held = {p["sym"] for p in st["positions"]}
    cands, decisions = [], []
    for s in sigs:
        key = f"{s['type']}|{s['state']}|{s['flagged']}"
        g = gates.get(key) or dict(ok=False, level="표본 부족", n=0)
        d = dict(sym=s["sym"], type=s["type"], state=s["state"], flagged=int(s["flagged"]), n=g.get("n", 0), meanR=g.get("meanR"), lb=g.get("lb"), level=g.get("level"), grade=g.get("grade"))
        if s["sym"] in held:
            d.update(decision="건너뜀", why="이미 보유 중")
        elif not g.get("ok"):
            d.update(decision="관찰만", why="표본 부족" if g.get("level") == "표본 부족" else ("최근 1년 성적이 마이너스" if (g.get("lb") or -1) > 0 else "과거 성적이 플러스라고 보기 어려움"))
        else:
            cands.append((0 if g["grade"] == "A" else 1, -(g.get("lb") or 0), s.get("range60") if s.get("range60") is not None and np.isfinite(s.get("range60")) else 9, s, g, d))
        decisions.append(d)
    n_alt = len(st["positions"])
    for _, _, _, s, g, d in sorted(cands, key=lambda x: x[:3]):
        b = px(s["sym"])
        sref = stop_ref(s["sym"])
        risk = clamp_risk(b[3], sref) if b is not None else 0.1
        frac = P["risk_a"] if g["grade"] == "A" else P["risk_b"]
        value = min(eq * P["alt_cap"], eq * frac / risk)
        if n_alt >= P["max_alt"]:
            d.update(decision="건너뜀", why="알트 자리 12개가 다 참")
        elif alt_val + value > eq * P["alt_total"]:
            d.update(decision="건너뜀", why="알트 비중 한도 55%")
        elif cash_avail - value < eq * P["reserve"]:
            d.update(decision="건너뜀", why="현금 부족")
        else:
            st["pending"].append(dict(kind="alt_buy", sym=s["sym"], type=s["type"], state=s["state"], flagged=int(s["flagged"]), grade=g["grade"], riskFrac=frac, stopRef=None if sref is None or not np.isfinite(sref) else float(sref), signalDate=date,
                                      cell=dict(n=g.get("n"), meanR=g.get("meanR"), lb=g.get("lb"), win=g.get("win"), level=g.get("level"))))
            d.update(decision="매수 예약", why=f"{g['grade']}등급, 다음 날 시가", plan=round(value, 2))
            n_alt += 1
            alt_val += value
            cash_avail -= value
    st["decisions"] = decisions
    st["market"] = dict(date=date, state=mkt["state"], btc_rsi=None if not np.isfinite(mkt["btc_rsi"]) else round(float(mkt["btc_rsi"]), 1), btc_d50=None if not np.isfinite(mkt["btc_d50"]) else round(float(mkt["btc_d50"]), 4),
                        mkt_brk=round(float(mkt["mkt_brk"]), 4), names=int(mkt["names"]))
    st["closeDone"] = date


def make_access(coins, t):
    """과거 자료의 t번째 날에 대한 접근 함수들"""
    def px(sym):
        c = coins.get(sym)
        if c is None:
            return None
        i = c["pos"][t]
        return None if i < 0 else (c["o"][i], c["h"][i], c["l"][i], c["c"][i])

    def ended(sym):
        c = coins.get(sym)
        return c is None or t > c["last_t"]

    def major_sig(sym):
        c = coins.get(sym)
        if c is None or c["pos"][t] < 0:
            return None
        i = c["pos"][t]
        if c["brk20"][i] == 1:
            return "enter"
        if np.isfinite(c["lo10"][i]) and c["c"][i] < c["lo10"][i]:
            return "exit"
        return None

    def stop_ref(sym):
        c = coins.get(sym)
        if c is None or c["pos"][t] < 0:
            return None
        return c["stop_ref"][c["pos"][t]]

    return px, ended, major_sig, stop_ref


def major_long_now(c):
    """메이저 규칙을 처음부터 돌려 지금 보유 상태인지 (실시간 장부를 처음 만들 때만 쓴다)"""
    long = False
    for i in range(len(c["c"])):
        if not long and c["brk20"][i] == 1:
            long = True
        elif long and np.isfinite(c["lo10"][i]) and c["c"][i] < c["lo10"][i]:
            long = False
    return long


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)  # 반올림하지 않는다 — 장부를 읽고 쓸 때마다 수량이 조금씩 달라지면 안 된다
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}", "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=1)


def load(name, default=None):
    try:
        with open(f"{OUT}/{name}", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default
