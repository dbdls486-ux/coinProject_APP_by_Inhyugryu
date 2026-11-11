#!/usr/bin/env python3
"""
급등 예측 방법론 — 지표 100여 개를 같은 잣대로 채점하고, 섞어서도 본다.

  목표: 오늘 종가에 고른 코인을 내일 시가에 사서 7일 보유.
        ① 7일 안에 매수가 대비 +30% 이상 오른 적이 있는가 (급등 적중)
        ② 수수료·슬리피지를 뺀 실제 수익
        ③ 같은 날 비슷하게 출렁이는 코인 평균 대비 초과수익

  1. 개별 지표: 날짜별로 지표 순위와 이후 수익 순위의 상관(IC), 상위 10% 의 급등 적중 배수·수익
  2. 신호형 지표(골든크로스·피보나치·일목 등): 신호가 뜬 날의 급등 확률·수익
  3. 신호 2개 조합
  4. 섞은 점수: (a) 학습 기간에 통한 지표의 합성 점수 (b) 기계학습 모델 — 매년 그 전 데이터로만 학습해 그 해에 적용
  5. 익절·손절 조합
  6. BTC·ETH 한 종목에 신호를 적용했을 때

  고르는 기간: ~2023-12-31 / 확인 기간: 2024-01-01~
  무작위 가짜 지표 5개를 같이 채점해 "우연히 나오는 수준"을 표시한다.
"""
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import BINARY, LABEL, THRESH, features, targets  # noqa: E402

warnings.filterwarnings("ignore")
DATA = os.environ.get("LAB_DATA", "lab/.data")
OUT = os.environ.get("LAB_OUT", "lab/results")
SPLIT = os.environ.get("LAB_SPLIT", "2024-01-01")
H = 7
K = 5  # 하루에 고르는 종목 수
MIN_AGE = 60
MIN_NAMES = 20
RNG = np.random.default_rng(20261111)
STABLE = set("USDT USDC USDS USDE DAI USD1 PYUSD FDUSD TUSD BUSD RLUSD USDP GUSD EURC USDQ USDG EUR EURI AEUR PAXG XUSD".split())
MARKETS = {
    "bithumb": dict(label="빗썸 원화마켓", min_value=1e9, cost=0.0025 + 0.003, tz=9),
    "binance": dict(label="바이낸스 USDT마켓", min_value=1e6, cost=0.001 + 0.0015, tz=0),
}
# 달러·유로 등에 값이 묶인 자산과 BTC·ETH·SOL 을 그대로 따라가는 포장 토큰 — 알트 종목 고르기에서 뺀다.
# (하락장에서는 '안 움직이는 자산'이 또래보다 잘한 것으로 잡혀 조용한 코인·초과수익 모델 성적이 부풀려진다)
# FRAX 는 넣지 않는다: 거래소의 FRAX 는 옛 FXS(프랙스 셰어)가 이름을 바꾼 일반 코인이다. 진짜 고정 가격 자산은 아래 변동폭 기준으로 걸러진다.
PEGGED = set("AUD GBP JPY TRY BRL SUSD UST USTC BFUSD USDSB PAX BKRW BVND VAI USDD LUSD USDJ USDN CUSD WBTC WBETH BETH BNSOL STETH WSTETH CBBTC WBNB BTCB".split())
MIN_ATR = 0.007  # 하루 변동폭이 0.7% 도 안 되는 날은 사실상 고정 가격 자산으로 보고 뺀다
RULE_COLS = ["d_sma20", "d_sma50", "d_sma200", "ma_align", "macd_hist", "ichi_cloud", "brk20", "brk60", "newlow20", "rsi14", "bb_pctb", "fib_zone"]
PLACEBO = [f"noise{i}" for i in range(1, 6)]
MARKET_FEATS = ["btc_d_sma50", "btc_roc7", "btc_roc28", "btc_rsi14", "breadth50", "mkt_ret1", "mkt_ret7"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hac_t(x, lag=H):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return float("nan")
    d = x - x.mean()
    v = d @ d / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return float(x.mean() / np.sqrt(max(v, 1e-18) / n))


# ----------------------------------------------------------------------------------------
# 자료 읽기
# ----------------------------------------------------------------------------------------
def load_market(mkt, cfg):
    today = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=cfg["tz"])).strftime("%Y-%m-%d")
    coins = {}
    for path in glob.glob(f"{DATA}/{mkt}/*.json"):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        rows = [r for r in j["rows"] if r[0] < today]  # 오늘 미완성 봉 제외
        if len(rows) < 90:
            continue
        a = np.array([r[1:6] for r in rows], dtype=float)
        dates = [r[0] for r in rows]
        ratio = a[1:, 3] / a[:-1, 3]
        bad = np.where((ratio > 20) | (ratio < 0.05))[0]  # 액면 변경·토큰 교체로 보이는 봉 이전은 버린다
        if len(bad):
            a, dates = a[bad[-1] + 1 :], dates[bad[-1] + 1 :]
        if len(dates) < 90 or not np.all(np.isfinite(a[:, :4])) or (a[:, :4] <= 0).any():
            continue
        coins[j["sym"]] = dict(dates=dates, o=a[:, 0], h=a[:, 1], l=a[:, 2], c=a[:, 3], v=np.nan_to_num(a[:, 4]), name=j.get("name", j["sym"]), delisted=bool(j.get("delisted")))
    return coins


def close_series(coin):
    return pd.Series(coin["c"], index=coin["dates"])


def prev_dates(dates):
    return (pd.to_datetime(dates) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def build(mkt, cfg, raw):
    """지표·목표값을 붙인 긴 표 (종목×날짜)"""
    coins = raw[mkt]
    other = raw["binance" if mkt == "bithumb" else "bithumb"]
    btc = close_series(coins["BTC"])
    usdt = close_series(raw["bithumb"]["USDT"]) if "USDT" in raw["bithumb"] else None
    all_dates = sorted({d for c in coins.values() for d in c["dates"]})
    tix = {d: i for i, d in enumerate(all_dates)}
    parts = []
    for sym, c in coins.items():
        dates = c["dates"]
        extra = {}
        if usdt is not None and sym in other:
            oc = close_series(other[sym])
            if mkt == "bithumb":  # 빗썸 마감(한국 자정) 시점에 확정된 바이낸스 종가는 전날 것
                extra["prem"] = c["c"] / (oc.reindex(prev_dates(dates)).to_numpy() * usdt.reindex(dates).to_numpy()) - 1
            else:
                extra["prem"] = oc.reindex(dates).to_numpy() / (c["c"] * usdt.reindex(dates).to_numpy()) - 1
        fpath = f"{DATA}/funding/{sym}.json"
        if os.path.exists(fpath):
            with open(fpath, encoding="utf-8") as f:
                fr = pd.Series(dict(json.load(f)["rows"]))
            extra["fund"] = fr.reindex(prev_dates(dates) if mkt == "bithumb" else dates).to_numpy()
        F = features(c["o"], c["h"], c["l"], c["c"], c["v"], btc.reindex(dates).ffill().to_numpy(), extra)
        c["F"] = F[RULE_COLS].astype("float32")  # 규칙 실험(2차 F·H)에서 다시 쓴다
        Tg = targets(c["o"], c["h"], c["l"], c["c"], cfg["cost"], H)
        F["sym"] = sym
        F["date"] = dates
        F["t"] = [tix[d] for d in dates]
        F["bar"] = np.arange(len(dates))
        for col in Tg.columns:
            F[col] = Tg[col].to_numpy()
        parts.append(F)
    df = pd.concat(parts, ignore_index=True)
    feats = [c for c in LABEL if c in df.columns]

    # 시장 상태 (모든 코인에 같은 값) — 기계학습 모델에만 쓴다
    b = df[df.sym == "BTC"].set_index("t")
    live = (df.bar >= MIN_AGE) & (df.val20 >= cfg["min_value"])
    day = df[live].groupby("t").agg(breadth50=("d_sma50", lambda s: float((s > 0).mean())), mkt_ret1=("roc1", "median"), mkt_ret7=("roc7", "median"))
    df["btc_d_sma50"] = df.t.map(b.d_sma50)
    df["btc_roc7"] = df.t.map(b.roc7)
    df["btc_roc28"] = df.t.map(b.roc28)
    df["btc_rsi14"] = df.t.map(b.rsi14)
    for col in day.columns:
        df[col] = df.t.map(day[col])
    for p in PLACEBO:
        df[p] = RNG.standard_normal(len(df)).astype("float32")

    # 거래 가능 종목만 남긴다
    named = df.sym.isin(STABLE | PEGGED)
    flat = live & ~named & (df.sym != "BTC") & df.fwd.notna() & ~(df.atr_pct >= MIN_ATR)
    excluded = dict(byName=sorted(set(df.sym[named & live])), byFlatPrice={k: int(v) for k, v in df.sym[flat].value_counts().head(30).items()})
    ok = live & ~named & (df.sym != "BTC") & df.fwd.notna() & (df.atr_pct >= MIN_ATR)
    df = df[ok].copy()
    n_day = df.groupby("t").sym.transform("count")
    df = df[n_day >= MIN_NAMES].copy()

    # 날짜별 순위(0~1), 변동성 5분위, 목표값
    cols = feats + PLACEBO
    rk = df.groupby("t")[cols].rank(pct=True).astype("float32")
    rk.columns = [f"r_{c}" for c in cols]
    df = pd.concat([df, rk], axis=1)
    df["bucket"] = np.minimum((df["r_atr_pct"].fillna(0.5) * 5).astype(int), 4)
    df["pump"] = (df.mx >= 0.30).astype(float)
    df["ex"] = df.fwd - df.groupby(["t", "bucket"]).fwd.transform("mean")
    # 순위 기준 목표값은 변동성 구간 안에서 매긴다. 전체에서 매기면 출렁임이 큰 코인이
    # 평균보다 아래에 있는 날이 많다는 이유만으로(수익 분포가 한쪽으로 치우쳐서) 변동성 지표가 통하는 것처럼 보인다.
    # (순위 − 0.5) / 개수 로 매겨야 구간마다 종목 수가 달라도 평균이 정확히 0.5 가 된다
    g = df.groupby(["t", "bucket"]).fwd
    df["yrank"] = (g.rank(method="average") - 0.5) / g.transform("count")
    df["win5"] = (df.yrank >= 0.8).astype(float)
    # 4주 보유 버전: 28일 안에 +50% / 28일 초과수익 상위 20% (최근 28일은 결과가 아직 없어 NaN)
    g28 = df.groupby(["t", "bucket"]).fwd28
    df["pump28"] = np.where(df.mx28.notna(), (df.mx28 >= 0.50).astype(float), np.nan)
    df["ex28"] = df.fwd28 - g28.transform("mean")
    df["win28"] = np.where(df.fwd28.notna(), ((g28.rank(method="average") - 0.5) / g28.transform("count") >= 0.8).astype(float), np.nan)
    df["is_test"] = df.date >= SPLIT
    return df.reset_index(drop=True), feats, all_dates, excluded


# ----------------------------------------------------------------------------------------
# 1. 개별 지표
# ----------------------------------------------------------------------------------------
def scan_features(df, feats):
    cols = feats + PLACEBO
    R = df[[f"r_{c}" for c in cols]].to_numpy(dtype="float32")
    R = np.where(np.isfinite(R), R, 0.5)
    y = df.yrank.to_numpy()
    pump = df.pump.to_numpy()
    fwd = df.fwd.to_numpy()
    ex = df.ex.to_numpy()
    test = df.is_test.to_numpy()
    order = np.argsort(df.t.to_numpy(), kind="stable")
    tt = df.t.to_numpy()[order]
    bounds = np.flatnonzero(np.r_[True, tt[1:] != tt[:-1], True])
    nF = len(cols)
    ic, is_test_day = [], []
    top = {k: np.zeros((2, nF)) for k in ("n", "pump", "fwd")}
    bot = {k: np.zeros((2, nF)) for k in ("n", "pump", "fwd")}
    top_ex_daily = [[[] for _ in range(nF)] for _ in range(2)]
    base = np.zeros((2, 2))
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        X = R[idx]
        yy = y[idx]
        Xc = X - X.mean(axis=0)
        yc = yy - yy.mean()
        den = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())
        ic.append(np.where(den > 0, (Xc * yc[:, None]).sum(axis=0) / np.where(den > 0, den, 1), np.nan))
        p = int(test[idx[0]])
        is_test_day.append(p)
        base[p] += [len(idx), pump[idx].sum()]
        hi = X >= 0.9
        lo = X <= 0.1
        for grp, m in ((top, hi), (bot, lo)):
            grp["n"][p] += m.sum(axis=0)
            grp["pump"][p] += (m * pump[idx][:, None]).sum(axis=0)
            grp["fwd"][p] += (m * fwd[idx][:, None]).sum(axis=0)
        cnt = hi.sum(axis=0)
        mean_ex = (hi * ex[idx][:, None]).sum(axis=0) / np.where(cnt > 0, cnt, 1)
        for f in range(nF):
            if cnt[f] > 0:
                top_ex_daily[p][f].append(mean_ex[f])
    ic = np.array(ic)
    is_test_day = np.array(is_test_day)
    rows = []
    for f, name in enumerate(cols):
        r = dict(feature=name, label=LABEL.get(name, "무작위 가짜 지표"), placebo=name in PLACEBO)
        for p, key in ((0, "train"), (1, "test")):
            x = ic[is_test_day == p, f]
            br = base[p, 1] / max(base[p, 0], 1)
            r[key] = dict(
                ic=float(np.nanmean(x)),
                ic_t=hac_t(x),
                top_lift=float(top["pump"][p, f] / max(top["n"][p, f], 1) / br) if br else None,
                bot_lift=float(bot["pump"][p, f] / max(bot["n"][p, f], 1) / br) if br else None,
                top_net=float(top["fwd"][p, f] / max(top["n"][p, f], 1)),
                bot_net=float(bot["fwd"][p, f] / max(bot["n"][p, f], 1)),
                top_ex=float(np.mean(top_ex_daily[p][f])) if top_ex_daily[p][f] else None,
                top_ex_t=hac_t(top_ex_daily[p][f]),
            )
        # 학습 기간 앞·뒤 절반의 부호가 같은가 (안정성)
        xt = ic[is_test_day == 0, f]
        half = len(xt) // 2
        r["stable"] = bool(np.sign(np.nanmean(xt[:half])) == np.sign(np.nanmean(xt[half:])))
        rows.append(r)
    return rows, dict(train=float(base[0, 1] / max(base[0, 0], 1)), test=float(base[1, 1] / max(base[1, 0], 1)))


# ----------------------------------------------------------------------------------------
# 2·3. 신호형 지표와 조합
# ----------------------------------------------------------------------------------------
def signal_columns(df):
    S = {}
    for b in BINARY:
        S[b] = (LABEL[b], (df[b] > 0.5).to_numpy())
    for key, (col, op, thr, label) in THRESH.items():
        x = df[col].to_numpy()
        S[key] = (label, (x < thr) if op == "<" else (x > thr))
    S["low_price"] = ("가격 단위 하위 20% (동전주)", (df["r_price_log"] < 0.2).to_numpy())
    return S


def signal_stats(df, mask):
    out = {}
    for p, key in ((False, "train"), (True, "test")):
        per = (df.is_test == p).to_numpy()
        m = mask & per
        n = int(m.sum())
        if n < 30:
            out[key] = dict(n=n)
            continue
        sub = df.loc[m, ["t", "ex", "fwd", "pump", "tp30sl10"]]
        base = float(df.pump.to_numpy()[per].mean())
        daily = sub.groupby("t").ex.mean().to_numpy()
        out[key] = dict(n=n, pump=float(sub.pump.mean()), lift=float(sub.pump.mean() / base) if base else None, ex=float(daily.mean()), t=hac_t(daily), net=float(sub.fwd.mean()), median=float(sub.fwd.median()), win=float((sub.fwd > 0).mean()), tpsl=float(sub.tp30sl10.mean()))
    return out


def scan_signals(df):
    S = signal_columns(df)
    singles = [dict(id=k, label=lab, **signal_stats(df, m)) for k, (lab, m) in S.items()]
    keys = list(S)
    pairs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            m = S[keys[i]][1] & S[keys[j]][1]
            if m.sum() < 400:
                continue
            st = signal_stats(df, m)
            if st["train"].get("n", 0) >= 200 and st["test"].get("n", 0) >= 200:
                pairs.append(dict(id=f"{keys[i]}+{keys[j]}", label=f"{S[keys[i]][0]} + {S[keys[j]][0]}", **st))
    return singles, pairs


def trim_pairs(pairs):
    ok = [p for p in pairs if p["train"].get("t") is not None and p["test"].get("t") is not None and np.isfinite(p["train"]["t"]) and np.isfinite(p["test"]["t"])]
    best = sorted(ok, key=lambda p: -min(p["train"]["t"], p["test"]["t"]))[:30]
    worst = sorted(ok, key=lambda p: max(p["train"]["t"], p["test"]["t"]))[:20]
    return dict(best=best, worst=worst)


# ----------------------------------------------------------------------------------------
# 4. 섞은 점수
# ----------------------------------------------------------------------------------------
def composite(df, scan):
    """고르는 기간에 통한 지표만(|t|≥3, 앞·뒤 절반 부호 일치) 방향을 맞춰 평균"""
    chosen = [r for r in scan if not r["placebo"] and r["stable"] and abs(r["train"]["ic_t"]) >= 3]
    chosen.sort(key=lambda r: -abs(r["train"]["ic_t"]))
    chosen = chosen[:25]
    if not chosen:
        return None, []
    s = np.zeros(len(df))
    for r in chosen:
        s += np.sign(r["train"]["ic"]) * (df[f"r_{r['feature']}"].fillna(0.5).to_numpy() - 0.5)
    return s / len(chosen), [dict(feature=r["feature"], label=r["label"], sign=int(np.sign(r["train"]["ic"])), t=r["train"]["ic_t"]) for r in chosen]


def walk_forward(df, feats, target, embargo=H + 2, importance=True):
    """매년 1월 1일 이전 자료로만 학습해 그 해를 예측. 학습 자료의 마지막 며칠(보유 기간+2일)은 목표값이 그 해와 겹치므로 버린다."""
    X_cols = [f"r_{c}" for c in feats] + MARKET_FEATS
    X = df[X_cols].to_numpy(dtype="float32")
    y = df[target].to_numpy()
    years = df.date.str[:4].astype(int).to_numpy()
    t = df.t.to_numpy()
    pred = np.full(len(df), np.nan)
    last_model = None
    for yr in range(max(2021, years.min() + 1), years.max() + 1):
        te = years == yr
        if not te.any():
            continue
        tr = (years < yr) & (t <= t[te].min() - embargo) & np.isfinite(y)
        if tr.sum() < 30000:
            continue
        # 학습 구간에 값이 거의 없는 지표(예: 그 시기에 없던 펀딩비)는 뺀다
        use = np.flatnonzero(np.isfinite(X[tr]).sum(axis=0) >= 500)
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=7)
        m.fit(X[tr][:, use], y[tr])
        pred[te] = m.predict_proba(X[te][:, use])[:, 1]
        last_model = (m, te, use)
        log(f"    {target} {yr}: 학습 {int(tr.sum()):,}행 → 예측 {int(te.sum()):,}행")
    imp = []
    if importance and last_model is not None:
        # 마지막 해에서 지표를 하나씩 섞어 봤을 때 급등 적중이 얼마나 떨어지는가
        m, te, use = last_model
        idx = np.flatnonzero(te)
        idx = RNG.choice(idx, size=min(60000, len(idx)), replace=False)
        Xs, ys = X[idx][:, use], y[idx]
        X_cols = [X_cols[i] for i in use]

        def top_rate(Xm):
            p = m.predict_proba(Xm)[:, 1]
            cut = np.quantile(p, 0.95)
            return ys[p >= cut].mean()

        base = top_rate(Xs)
        for f, col in enumerate(X_cols):
            Xp = Xs.copy()
            Xp[:, f] = RNG.permutation(Xp[:, f])
            imp.append((col[2:] if col.startswith("r_") else col, float(base - top_rate(Xp))))
        imp.sort(key=lambda x: -x[1])
    return pred, imp[:15]


class Ladder:
    """하루에 K종목씩 골라 7일 들고 가는 계좌 (매일 자금의 1/7 을 새 종목에 넣는다)"""

    def __init__(self, coins, all_dates, cost):
        self.dates = all_dates
        self.syms = sorted(coins)
        ci = {s: i for i, s in enumerate(self.syms)}
        T, N = len(all_dates), len(self.syms)
        C = np.full((T, N), np.nan)
        O = np.full((T, N), np.nan)
        tix = {d: i for i, d in enumerate(all_dates)}
        for s, c in coins.items():
            rows = [tix[d] for d in c["dates"]]
            C[rows, ci[s]] = c["c"]
            O[rows, ci[s]] = c["o"]
        self.Roc = C / O - 1
        Cf = pd.DataFrame(C).ffill().to_numpy()
        self.Rcc = np.vstack([np.full((1, N), np.nan), Cf[1:] / Cf[:-1] - 1])
        self.Rcc[~np.isfinite(C)] = 0.0
        self.ci, self.cost, self.T = ci, cost, T

    def run(self, picks, hold=H):
        port = np.zeros(self.T)
        for s, syms in picks.items():
            cols = [self.ci[x] for x in syms]
            for d in range(s + 1, min(s + hold, self.T - 1) + 1):
                r = np.nan_to_num(self.Roc[d, cols] if d == s + 1 else self.Rcc[d, cols])
                adj = (-self.cost if d == s + 1 else 0.0) + (-self.cost if d == s + hold else 0.0)
                port[d] += (r + adj).mean() / hold
        return port

    def stats(self, port, a, b):
        r = port[a : b + 1]
        eq = np.cumprod(1 + r)
        days = len(r)
        mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
        sd = r.std()
        return dict(cagr=float(eq[-1] ** (365 / days) - 1), mdd=mdd, sharpe=float(r.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0)

    def yearly(self, port, a, b):
        out = {}
        yrs = np.array([d[:4] for d in self.dates])
        for y in sorted(set(yrs[a : b + 1])):
            m = np.zeros(self.T, bool)
            m[a : b + 1] = yrs[a : b + 1] == y
            out[y] = float(np.prod(1 + port[m]) - 1)
        return out


def eval_score(df, score, lad, name, label, need=None):
    """점수 상위 K종목을 매일 샀을 때 — 연도별·기간별 성적"""
    s = pd.Series(score, index=df.index)
    ok = s.notna() if need is None else (s.notna() & need)
    sub = df[ok]
    rank = s[ok].groupby(sub.t).rank(ascending=False, method="first")
    picks = sub[rank <= K]
    if len(picks) < 200:
        return None
    out = dict(id=name, label=label, periods={}, topPicks={key: {k: int(v) for k, v in picks.sym[m].value_counts().head(12).items()} for key, m in (("train", ~picks.is_test), ("test", picks.is_test))})
    port = lad.run({int(t): list(g) for t, g in picks.groupby("t").sym})
    tmin, tmax = int(sub.t.min()), int(sub.t.max()) + H
    split_t = int(sub.t[sub.is_test].min()) if sub.is_test.any() else tmax + 1
    for key, m, a, b in (("train", ~picks.is_test, tmin, split_t - 1), ("test", picks.is_test, split_t, min(tmax, lad.T - 1))):
        p = picks[m]
        if len(p) < 100 or b <= a:
            continue
        base = sub[sub.is_test == (key == "test")]
        daily = p.groupby("t").ex.mean().to_numpy()
        out["periods"][key] = dict(
            trades=int(len(p)), pump=float(p.pump.mean()), base=float(base.pump.mean()), net=float(p.fwd.mean()), median=float(p.fwd.median()), win=float((p.fwd > 0).mean()),
            ex=float(daily.mean()), t=hac_t(daily), universe_net=float(base.fwd.mean()),
            exits={k: dict(mean=float(p[k].mean()), median=float(p[k].median()), win=float((p[k] > 0).mean())) for k in ("tp30sl10", "tp50sl15", "tp20sl7", "sl10")},
            **lad.stats(port, a, b),
        )
    out["years"] = lad.yearly(port, tmin, min(tmax, lad.T - 1))
    return out


# ----------------------------------------------------------------------------------------
# 6. BTC·ETH 한 종목에 적용
# ----------------------------------------------------------------------------------------
def single_coin(raw_mkt, sym, cfg):
    c = raw_mkt.get(sym)
    if c is None:
        return []
    btc = close_series(raw_mkt["BTC"]).reindex(c["dates"]).ffill().to_numpy()
    F = features(c["o"], c["h"], c["l"], c["c"], c["v"], btc)
    C, O = pd.Series(c["c"]), pd.Series(c["o"])
    entry = O.shift(-1)
    F["f7"] = (C.shift(-7) / entry * (1 - cfg["cost"]) / (1 + cfg["cost"]) - 1).to_numpy()
    F["f28"] = (C.shift(-28) / entry * (1 - cfg["cost"]) / (1 + cfg["cost"]) - 1).to_numpy()
    F["date"] = c["dates"]
    F = F[(np.arange(len(F)) >= 250) & (F.date >= "2018-01-01")]
    sigs = {b: (LABEL[b], F[b] > 0.5) for b in BINARY if b != "ratio_high60"}
    for key, (col, op, thr, label) in THRESH.items():
        if col in ("prem5", "fund7", "rs28", "price_log"):
            continue
        sigs[key] = (label, (F[col] < thr) if op == "<" else (F[col] > thr))
    out = []
    for key, (label, m) in sigs.items():
        row = dict(id=key, label=label)
        for per, pm in (("train", F.date < SPLIT), ("test", F.date >= SPLIT)):
            on = F[m & pm]
            allp = F[pm]
            row[per] = dict(n=int(on.f7.notna().sum()), f7=float(on.f7.mean()) if len(on) else None, base7=float(allp.f7.mean()), f28=float(on.f28.mean()) if len(on) else None, base28=float(allp.f28.mean()), win7=float((on.f7 > 0).mean()) if len(on) else None)
        out.append(row)
    return out


# ----------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    raw = {m: load_market(m, cfg) for m, cfg in MARKETS.items() if os.path.isdir(f"{DATA}/{m}")}
    result = dict(generatedAt=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M"), horizon=H, picksPerDay=K, markets=[])
    for mkt, cfg in MARKETS.items():
        if mkt not in raw or "BTC" not in raw[mkt]:
            continue
        other = "binance" if mkt == "bithumb" else "bithumb"
        raw.setdefault(other, {})
        log(f"[{cfg['label']}] 지표 계산 ({len(raw[mkt])}종목)")
        df, feats, all_dates, excluded = build(mkt, cfg, raw)
        log(f"  이름으로 제외: {', '.join(excluded['byName']) or '없음'} / 가격이 거의 안 움직여 제외된 날이 있는 종목: {excluded['byFlatPrice']}")
        log(f"  표 {len(df):,}행 × 지표 {len(feats)}개 · {df.date.min()} ~ {df.date.max()}")
        scan, base = scan_features(df, feats)
        log("  개별 지표 채점 완료")
        singles, pairs = scan_signals(df)
        log(f"  신호 {len(singles)}개, 조합 {len(pairs)}개 채점 완료")
        lad = Ladder(raw[mkt], all_dates, cfg["cost"])

        scores = []
        comp, comp_feats = composite(df, scan)
        df["s_rand"] = RNG.random(len(df))
        cands = [("s_rand", "무작위로 5종목", df.s_rand.to_numpy()), ("s_vol", "변동성(ATR%) 가장 큰 5종목", df.r_atr_pct.to_numpy()), ("s_pumps", "최근 90일 급등 횟수 가장 많은 5종목", df.r_pumps90.to_numpy() + df.s_rand.to_numpy() * 1e-6)]
        # 고르는 기간 성적이 가장 좋았던 개별 지표 3개 (방향 포함)
        best = sorted([r for r in scan if not r["placebo"] and r["stable"]], key=lambda r: -abs(r["train"]["ic_t"]))[:3]
        for r in best:
            sgn = np.sign(r["train"]["ic"])
            cands.append((f"s_{r['feature']}", f"개별 지표: {r['label']} {'높은' if sgn > 0 else '낮은'} 순", sgn * df[f"r_{r['feature']}"].to_numpy()))
        if comp is not None:
            cands.append(("s_comp", f"합성 점수 (고르는 기간에 통한 지표 {len(comp_feats)}개 평균)", comp))
        ml_pump, imp_pump = walk_forward(df, feats, "pump")
        ml_ret, imp_ret = walk_forward(df, feats, "win5")
        cands.append(("s_ml_pump", "기계학습: 급등 확률 예측", ml_pump))
        cands.append(("s_ml_ret", "기계학습: 초과수익 상위 20% 예측", ml_ret))
        if np.isfinite(ml_pump).any():
            both = pd.Series(ml_pump).rank(pct=True).to_numpy() + pd.Series(ml_ret).rank(pct=True).to_numpy()
            cands.append(("s_ml_both", "기계학습: 두 모델 평균", both))
        has_ml = np.isfinite(ml_pump)
        # 반대쪽: 조용한 코인, 고점 근처 코인
        calm = (df.bucket <= 2).to_numpy()
        cands.append(("s_quiet", "변동성(ATR%) 가장 작은 5종목", -df.r_atr_pct.to_numpy()))
        cands.append(("s_ml_pump_low", "기계학습: 급등 확률이 가장 낮은 5종목", -ml_pump))
        cands.append(("s_ml_ret_calm", "기계학습 초과수익 모델, 변동성 하위 60% 안에서만", np.where(calm, ml_ret, np.nan)))
        cands.append(("s_near_high", "250일 고점에 가장 가까운 5종목", df.r_d_high250.to_numpy()))
        cands.append(("s_ml_ret_high", "기계학습 초과수익 모델, 250일 고점 10% 이내에서만", np.where((df.d_high250 > -0.1).to_numpy(), ml_ret, np.nan)))
        for sid, label, sc in cands:
            r = eval_score(df, sc, lad, sid, label)
            if r:
                scores.append(r)
            # 기계학습 예측이 있는 기간(매년 다시 학습한 구간)만 따로 — 같은 날짜끼리 비교
            r2 = eval_score(df, sc, lad, sid + "@oos", label, need=pd.Series(has_ml, index=df.index))
            if r2:
                scores.append(r2)
        log("  섞은 점수 평가 완료")
        from round2 import round2

        r2 = round2(df, feats, raw[mkt], lad, cfg, dict(cands=cands, ml_pump=ml_pump, ml_ret=ml_ret, has_ml=has_ml, skip=STABLE | PEGGED), walk_forward, hac_t, log, SPLIT)
        oos_first = df.date[has_ml].min()[:4] if has_ml.any() else None
        log("  2차 실험 완료")
        # BTC 그냥 보유 (같은 구간)
        btc = raw[mkt]["BTC"]
        bs = pd.Series(btc["c"], index=btc["dates"])
        hold = {}
        for key, a, b in (("train", df.date[~df.is_test].min(), df.date[~df.is_test].max()), ("test", df.date[df.is_test].min(), df.date[df.is_test].max())):
            x = bs[(bs.index >= a) & (bs.index <= b)]
            eq = x / x.iloc[0]
            hold[key] = dict(cagr=float(eq.iloc[-1] ** (365 / len(x)) - 1), mdd=float((eq / eq.cummax() - 1).min()))
        result["markets"].append(dict(
            market=mkt, label=cfg["label"], rows=int(len(df)), coins=int(df.sym.nunique()), cost=cfg["cost"], dates=[df.date.min(), df.date.max()], base=base,
            scan=scan, signals=singles, pairCount=len(pairs), pairs=trim_pairs(pairs), composite=comp_feats, importance=dict(pump=imp_pump, ret=imp_ret), scores=scores, btcHold=hold,
            single={s: single_coin(raw[mkt], s, cfg) for s in ("BTC", "ETH")}, round2=r2, excluded=excluded,
            oosTrain=None if oos_first is None else (oos_first if oos_first == str(int(SPLIT[:4]) - 1) else f"{oos_first}~{int(SPLIT[:4]) - 1}"),
        ))
        del df
    os.makedirs(OUT, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else round(float(o), 5)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    result = clean(result)
    with open(f"{OUT}/indicators.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from report import render

    with open(f"{OUT}/INDICATORS.md", "w", encoding="utf-8") as f:
        f.write(render(result))
    log(f"끝 ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
