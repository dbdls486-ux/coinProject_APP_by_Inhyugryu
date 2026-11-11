"""
2차 실험 — 1차 결과("급등 확률은 맞히는데 사면 손해")에 흔히 붙는 반론을 하나씩 확인한다.

  A. 상승장일 때만 사면?          (BTC 50일선 / 알트 절반 이상 50일선 위 / 알트 중앙값 7일 상승)
  B. 모델이 아주 확신할 때만 사면?  (예측 확률 구간별 실제 급등 비율·수익)
  C. 하루에 1종목만 / 20종목까지 사면?
  D. 익절·손절·보유 기간을 바꾸면?  (3·7·14일 × 익절 5가지 × 손절 5가지 = 75조합)
  E. 4주 보유로 길게 보면?          (28일 안에 +50% 를 맞히는 모델을 따로 학습)
  F. 같은 지표를 메이저 코인에 추세 규칙으로 쓰면?
  G. 거래대금 상위 30% 종목 안에서만 고르면?
  H. 추세·역추세 규칙을 거래대금 순위 구간별(1~5위 / 6~30위 / 31~100위 / 그 아래) 바구니에 적용하면?
"""
import numpy as np
import pandas as pd

from features import features, path_outcomes

K = 5
H = 7
LONG = 28
REGIMES = [
    ("btc50", "BTC가 50일선 위", lambda d: (d.btc_d_sma50 > 0).to_numpy()),
    ("breadth", "알트 절반 이상이 50일선 위", lambda d: (d.breadth50 >= 0.5).to_numpy()),
    ("mkt7", "알트 중앙값이 7일간 상승", lambda d: (d.mkt_ret7 > 0).to_numpy()),
    ("both", "BTC 50일선 위 + 알트 절반 이상 50일선 위", lambda d: ((d.btc_d_sma50 > 0) & (d.breadth50 >= 0.5)).to_numpy()),
]
HZ = (3, 7, 14)
TPS = (0, 0.10, 0.20, 0.30, 0.50)
SLS = (0, 0.05, 0.10, 0.15, 0.20)
COMBOS = [(tp, sl) for tp in TPS for sl in SLS]
MAJORS = ["BTC", "ETH", "XRP", "SOL", "DOGE"]


def picks_of(df, score, need=None, k=K):
    s = pd.Series(score, index=df.index)
    ok = s.notna() if need is None else (s.notna() & need)
    sub = df[ok]
    rank = s[ok].groupby(sub.t).rank(ascending=False, method="first")
    return sub[rank <= k]


def trade_stats(p, base, hac_t, fwd="fwd", ex="ex", pump="pump", lag=H):
    if len(p) < 30:
        return dict(trades=int(len(p)))
    daily = p.groupby("t")[ex].mean().to_numpy()
    # 날 수가 적으면(100일 미만) t 값이 부풀려지기 쉬워 표시하지 않는다 (신호 없는 가짜 시세에서 확인)
    return dict(
        trades=int(len(p)), days=int(p.t.nunique()), pump=float(p[pump].mean()), base=float(base[pump].mean()) if len(base) else None,
        net=float(p[fwd].mean()), median=float(p[fwd].median()), win=float((p[fwd] > 0).mean()), ex=float(daily.mean()), t=hac_t(daily, lag) if len(daily) >= 100 else None,
        universe_net=float(base[fwd].mean()) if len(base) else None,
    )


def windows(sub, lad, hold=H):
    """계좌 성적을 재는 구간 (t 번호): 고르는 기간 / 확인 기간"""
    tmin, tmax = int(sub.t.min()), min(int(sub.t.max()) + hold, lad.T - 1)
    split_t = int(sub.t[sub.is_test].min()) if sub.is_test.any() else tmax + 1
    return dict(train=(tmin, split_t - 1), test=(split_t, tmax))


def account(lad, picks, win, hold=H):
    port = lad.run({int(t): list(g) for t, g in picks.groupby("t").sym}, hold)
    out = {}
    for key, (a, b) in win.items():
        if b - a > 30:
            out[key] = lad.stats(port, a, b)
    return out


# ----------------------------------------------------------------------------------------
# A. 장세별
# ----------------------------------------------------------------------------------------
def regime_split(df, methods, lad, hac_t, has_ml):
    need = pd.Series(has_ml, index=df.index)
    uni = df[has_ml]
    win = windows(uni, lad)
    out = []
    picks_cache = {mid: picks_of(df, sc, need) for mid, _, sc in methods}
    for rid, rlabel, fn in REGIMES:
        on = fn(df)
        day_on = pd.Series(on[uni.index.to_numpy()], index=uni.index).groupby(uni.t).first()
        day_test = uni.groupby("t").is_test.first()
        share = dict(train=float(day_on[~day_test].mean()), test=float(day_on[day_test].mean()))
        rows = []
        for mid, mlabel, _ in methods:
            picks = picks_cache[mid]
            p_on = on[picks.index.to_numpy()]
            u_on = on[uni.index.to_numpy()]
            cells = {}
            for per, pm, um in (("train", ~picks.is_test, ~uni.is_test), ("test", picks.is_test, uni.is_test)):
                for state, sm, usm in (("on", p_on, u_on), ("off", ~p_on, ~u_on)):
                    cells[f"{per}_{state}"] = trade_stats(picks[pm.to_numpy() & sm], uni[um.to_numpy() & usm], hac_t)
            rows.append(dict(id=mid, label=mlabel, cells=cells, account=account(lad, picks[p_on], win)))
        out.append(dict(id=rid, label=rlabel, share=share, rows=rows))
    return out


# ----------------------------------------------------------------------------------------
# B. 확신도별
# ----------------------------------------------------------------------------------------
def calibration(df, pred, edges, hac_t, target="pump"):
    ok = np.isfinite(pred)
    rows = []
    for per, pm in (("train", ~df.is_test.to_numpy()), ("test", df.is_test.to_numpy())):
        m = ok & pm
        d = df.loc[m, ["t", target, "fwd", "ex", "tp30sl10"]].copy()
        d["p"] = pred[m]
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = d[(d.p >= lo) & (d.p < hi)]
            r = dict(period=per, lo=lo, hi=hi, n=int(len(b)), days=int(b.t.nunique()))
            if len(b) >= 30:
                daily = b.groupby("t").ex.mean().to_numpy()
                r.update(pred=float(b.p.mean()), hit=float(b[target].mean()), net=float(b.fwd.mean()), median=float(b.fwd.median()), win=float((b.fwd > 0).mean()), ex=float(daily.mean()), t=hac_t(daily) if len(daily) >= 100 else None, tpsl=float(b.tp30sl10.mean()))
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------------------
# C. 하루에 고르는 종목 수
# ----------------------------------------------------------------------------------------
def k_sweep(df, methods, hac_t, has_ml):
    need = pd.Series(has_ml, index=df.index)
    uni = df[has_ml]
    out = []
    for mid, mlabel, sc in methods:
        for k in (1, 3, 5, 10, 20):
            picks = picks_of(df, sc, need, k)
            out.append(dict(id=mid, label=mlabel, k=k, train=trade_stats(picks[~picks.is_test], uni[~uni.is_test], hac_t), test=trade_stats(picks[picks.is_test], uni[uni.is_test], hac_t)))
    return out


# ----------------------------------------------------------------------------------------
# G. 거래대금이 큰 종목 안에서만 고르면
# ----------------------------------------------------------------------------------------
def liquid_only(df, methods, lad, hac_t, has_ml, cut=0.7):
    big = (df.r_val20_log >= cut).to_numpy() & has_ml
    need = pd.Series(big, index=df.index)
    uni = df[big]
    win = windows(uni, lad)
    rows = []
    for mid, mlabel, sc in methods:
        picks = picks_of(df, sc, need)
        rows.append(dict(id=mid, label=mlabel, train=trade_stats(picks[~picks.is_test], uni[~uni.is_test], hac_t), test=trade_stats(picks[picks.is_test], uni[uni.is_test], hac_t), account=account(lad, picks, win)))
    return dict(rows=rows, names=float(uni.groupby("t").sym.count().median()))


# ----------------------------------------------------------------------------------------
# D. 익절·손절·보유 기간
# ----------------------------------------------------------------------------------------
def exit_grid(coins, picks_by_method, cost):
    need = {}
    for _, picks in picks_by_method.values():
        for sym in picks.sym.unique():
            need[sym] = True
    cache = {}
    for sym in need:
        c = coins[sym]
        cache[sym] = {hz: path_outcomes(c["o"], c["h"], c["l"], c["c"], cost, hz, COMBOS) for hz in HZ}
    out = []
    for mid, (mlabel, picks) in picks_by_method.items():
        syms, bars, is_test = picks.sym.to_numpy(), picks.bar.to_numpy(), picks.is_test.to_numpy()
        cells = []
        for hz in HZ:
            M = np.vstack([cache[s][hz][b] for s, b in zip(syms, bars)])
            ok = np.isfinite(M).all(axis=1)
            st = {}
            for per, pm in (("train", ~is_test), ("test", is_test)):
                X = M[pm & ok]
                st[per] = (X.mean(axis=0), np.median(X, axis=0), (X > 0).mean(axis=0), len(X)) if len(X) >= 100 else None
            for j, (tp, sl) in enumerate(COMBOS):
                cell = dict(hz=hz, tp=tp, sl=sl)
                for per in ("train", "test"):
                    if st[per]:
                        cell[per] = dict(mean=float(st[per][0][j]), median=float(st[per][1][j]), win=float(st[per][2][j]), n=int(st[per][3]))
                cells.append(cell)
        full = [c for c in cells if "train" in c and "test" in c]
        best = max(full, key=lambda c: c["train"]["mean"]) if full else None
        out.append(dict(
            id=mid, label=mlabel, cells=cells, bestByTrain=best,
            positiveTest=int(sum(c["test"]["mean"] > 0 for c in full)), positiveTrain=int(sum(c["train"]["mean"] > 0 for c in full)),
            positiveBoth=int(sum(c["test"]["mean"] > 0 and c["train"]["mean"] > 0 for c in full)), combos=len(full),
        ))
    return out


# ----------------------------------------------------------------------------------------
# E. 4주 보유
# ----------------------------------------------------------------------------------------
def long_horizon(df, feats, lad, walk_forward, hac_t, log):
    pred_p, _ = walk_forward(df, feats, "pump28", embargo=LONG + 2, importance=False)
    pred_r, _ = walk_forward(df, feats, "win28", embargo=LONG + 2, importance=False)
    has = np.isfinite(pred_p) & df.fwd28.notna().to_numpy()
    if has.sum() < 5000:
        return None
    need = pd.Series(has, index=df.index)
    uni = df[has]
    win = windows(uni, lad, LONG)
    rnd = np.random.default_rng(28).random(len(df))
    methods = [
        ("rand", "무작위로 5종목", rnd), ("vol", "변동성(ATR%) 가장 큰 5종목", df.r_atr_pct.to_numpy()), ("quiet", "변동성(ATR%) 가장 작은 5종목", -df.r_atr_pct.to_numpy()),
        ("ml_pump28", "기계학습: 28일 안에 +50% 급등 확률", pred_p), ("ml_ret28", "기계학습: 28일 초과수익 상위 20% 예측", pred_r),
        ("near_high", "250일 고점에 가장 가까운 5종목", df.r_d_high250.to_numpy()),
    ]
    rows = []
    for mid, mlabel, sc in methods:
        picks = picks_of(df, sc, need)
        r = dict(id=mid, label=mlabel, account=account(lad, picks, win, LONG), topPicks={k: int(v) for k, v in picks.sym[picks.is_test].value_counts().head(12).items()})
        for per, pm, um in (("train", ~picks.is_test, ~uni.is_test), ("test", picks.is_test, uni.is_test)):
            r[per] = trade_stats(picks[pm], uni[um], hac_t, "fwd28", "ex28", "pump28", LONG)
        rows.append(r)
    return dict(rows=rows, base=dict(train=float(uni.pump28[~uni.is_test].mean()), test=float(uni.pump28[uni.is_test].mean())))


# ----------------------------------------------------------------------------------------
# F. 메이저 코인에 추세·역추세 규칙 적용
# ----------------------------------------------------------------------------------------
def _rules(F, C):
    f = lambda col: F[col].to_numpy()
    lo10 = C.rolling(10).min().shift(1).to_numpy()
    c = C.to_numpy()
    T, N = np.ones(len(c), bool), np.zeros(len(c), bool)
    return [
        ("HOLD", "그냥 보유", "기준", T, N, 0),
        ("MA50", "50일선 위에서만 보유", "추세", f("d_sma50") > 0, f("d_sma50") < 0, 0),
        ("MA200", "200일선 위에서만 보유", "추세", f("d_sma200") > 0, f("d_sma200") < 0, 0),
        ("GC50_200", "골든크로스(50·200일)에 사서 데드크로스에 팔기", "추세", f("d_sma200") > f("d_sma50"), f("d_sma200") < f("d_sma50"), 0),
        ("MA_ALIGN", "이평선 정배열일 때만 보유", "추세", f("ma_align") > 0.5, f("ma_align") < 0.5, 0),
        ("MACD", "MACD가 시그널 위일 때만 보유", "추세", f("macd_hist") > 0, f("macd_hist") < 0, 0),
        ("ICHI", "일목 구름 위에서 사서 구름 아래로 내려가면 팔기", "추세", f("ichi_cloud") > 0.5, f("ichi_cloud") < -0.5, 0),
        ("DON20", "20일 고점 돌파에 사서 10일 저점 이탈에 팔기", "추세", f("brk20") > 0.5, c < lo10, 0),
        ("DON60", "60일 고점 돌파에 사서 20일 저점 이탈에 팔기", "추세", f("brk60") > 0.5, f("newlow20") > 0.5, 0),
        ("RSI_MOM", "RSI 70 넘으면 사서 50 아래로 내려가면 팔기", "추세", f("rsi14") > 70, f("rsi14") < 50, 0),
        ("BB_UP", "볼린저 상단 돌파에 사서 20일선 아래로 내려가면 팔기", "추세", f("bb_pctb") > 1, f("d_sma20") < 0, 0),
        ("RSI_DIP", "RSI 30 아래에서 사서 50 위로 올라오면 팔기", "역추세", f("rsi14") < 30, f("rsi14") > 50, 0),
        ("BB_DIP", "볼린저 하단 아래에서 사서 20일선 회복하면 팔기", "역추세", f("bb_pctb") < 0, f("d_sma20") > 0, 0),
        ("FIB", "피보나치 0.382~0.618 눌림에서 사서 20일 고점 돌파 또는 28일 뒤 팔기", "역추세", f("fib_zone") > 0.5, f("brk20") > 0.5, 28),
    ]


def _feat(coins, c):
    if "F" in c:
        return c["F"]
    btc = pd.Series(coins["BTC"]["c"], index=coins["BTC"]["dates"]).reindex(c["dates"]).ffill().to_numpy()
    return features(c["o"], c["h"], c["l"], c["c"], c["v"], btc)


def _run_rule(o, c, ent, ext, cost, a, b, max_hold):
    """판단은 종가에, 체결은 다음 날 시가에. 구간 [a, b] 를 현금으로 시작한다."""
    cash, qty, held, buy, sell = 1.0, 0.0, 0, False, False
    eq = np.empty(b - a + 1)
    in_days = trades = 0
    for i in range(a, b + 1):
        if buy and qty == 0:
            qty, cash, held = cash * (1 - cost) / o[i], 0.0, 0
            trades += 1
        elif sell and qty > 0:
            cash, qty = qty * o[i] * (1 - cost), 0.0
        buy = sell = False
        eq[i - a] = cash + qty * c[i]
        if qty > 0:
            in_days += 1
            held += 1
            sell = bool(ext[i]) or (max_hold > 0 and held >= max_hold)
        else:
            buy = bool(ent[i])
    return eq, in_days / len(eq), trades


def _curve_stats(eq):
    days = len(eq)
    r = np.diff(np.r_[1.0, eq]) / np.r_[1.0, eq[:-1]]
    sd = r.std()
    return dict(cagr=float(eq[-1] ** (365 / days) - 1), mdd=float((eq / np.maximum.accumulate(eq) - 1).min()), sharpe=float(r.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0)


def majors_rules(coins, cost, split):
    per_rule = {}
    for sym in MAJORS:
        c = coins.get(sym)
        if c is None or len(c["dates"]) < 700:
            continue
        dates = np.array(c["dates"])
        F = _feat(coins, c)
        C = pd.Series(c["c"])
        start = max(250, int(np.searchsorted(dates, "2018-01-01")))
        sp = int(np.searchsorted(dates, split))
        spans = dict(train=(start, sp - 1), test=(sp, len(dates) - 1))
        for rid, label, kind, ent, ext, mh in _rules(F, C):
            R = per_rule.setdefault(rid, dict(id=rid, label=label, kind=kind, coins={}, _ret={"train": {}, "test": {}}))
            ent, ext = np.nan_to_num(ent.astype(float)) > 0.5, np.nan_to_num(ext.astype(float)) > 0.5
            for per, (a, b) in spans.items():
                if b - a < 200:
                    continue
                eq, expo, trades = _run_rule(c["o"], c["c"], ent, ext, cost, a, b, mh)
                R["coins"].setdefault(sym, {})[per] = dict(**_curve_stats(eq), exposure=float(expo), trades=int(trades))
                R["_ret"][per][sym] = pd.Series(np.diff(np.r_[1.0, eq]) / np.r_[1.0, eq[:-1]], index=dates[a : b + 1])
    out = []
    for R in per_rule.values():
        basket = {}
        for per, parts in R.pop("_ret").items():
            if len(parts) >= 2:
                r = pd.concat(parts, axis=1).mean(axis=1)  # 그날 자료가 있는 코인끼리 같은 비중
                basket[per] = dict(**_curve_stats(np.cumprod(1 + r.to_numpy())), coins=len(parts))
        R["basket"] = basket
        out.append(R)
    return out


# ----------------------------------------------------------------------------------------
# H. 같은 규칙을 거래대금 순위 구간별 바구니에 적용
# ----------------------------------------------------------------------------------------
TIERS = [("t1", "거래대금 1~5위", 1, 5), ("t2", "6~30위", 6, 30), ("t3", "31~100위", 31, 100), ("t4", "101위 이하", 101, 10**6)]
TIER_RULES = ("HOLD", "MA50", "MA200", "GC50_200", "MA_ALIGN", "MACD", "ICHI", "DON20", "DON60", "RSI_MOM", "BB_UP", "RSI_DIP", "BB_DIP", "FIB")
WARM = 250


def tier_rules(coins, cost, split, min_value, skip):
    """
    코인마다 규칙대로 굴린 계좌를 만들고, 날마다 전날 기준 20일 평균 거래대금 순위로 구간을 나눠 같은 비중으로 묶는다.
    구간이 바뀔 때의 재조정 비용은 넣지 않았다 (그냥 보유 바구니도 똑같이 뺐으므로 서로 비교는 공정하다).
    'BTC 50일선 위일 때만' 은 규칙의 매수를 BTC 가 50일선 위인 날로 제한하고, BTC 가 50일선 아래로 내려가면 판다.
    """
    all_dates = sorted({d for c in coins.values() for d in c["dates"]})
    tix = {d: i for i, d in enumerate(all_dates)}
    T = len(all_dates)
    syms = [s for s, c in coins.items() if s not in skip and len(c["dates"]) >= WARM + 80]
    N = len(syms)
    V = np.full((T, N), np.nan)
    btcF = _feat(coins, coins["BTC"])
    btc_up = pd.Series((btcF["d_sma50"].to_numpy() > 0), index=coins["BTC"]["dates"])
    ids = list(TIER_RULES) + [f"{r}+BTC" for r in ("MA50", "DON20", "BB_UP")]
    RR = {rid: np.full((T, N), np.nan, dtype="float32") for rid in ids}
    meta = {}
    for j, s in enumerate(syms):
        c = coins[s]
        rows = np.array([tix[d] for d in c["dates"]])
        v20 = pd.Series(c["v"]).rolling(20).mean().shift(1).to_numpy()
        V[rows[WARM:], j] = v20[WARM:]
        F = _feat(coins, c)
        up = btc_up.reindex(c["dates"]).fillna(False).to_numpy(dtype=bool)
        n = len(rows)
        for rid, label, kind, ent, ext, mh in _rules(F, pd.Series(c["c"])):
            if rid not in RR:
                continue
            meta[rid] = (label, kind)
            ent, ext = np.nan_to_num(ent.astype(float)) > 0.5, np.nan_to_num(ext.astype(float)) > 0.5
            variants = [(rid, ent, ext)]
            if f"{rid}+BTC" in RR:
                meta[f"{rid}+BTC"] = (label + " (BTC 50일선 위일 때만)", kind)
                variants.append((f"{rid}+BTC", ent & up, ext | ~up))
            for vid, e1, x1 in variants:
                eq, _, _ = _run_rule(c["o"], c["c"], e1, x1, cost, WARM, n - 1, mh)
                RR[vid][rows[WARM:], j] = np.diff(np.r_[1.0, eq]) / np.r_[1.0, eq[:-1]]
    ok = np.isfinite(V) & (V >= min_value)
    rank = np.where(ok, (-np.where(ok, V, -np.inf)).argsort(axis=1).argsort(axis=1) + 1, 0)
    dates = np.array(all_dates)
    first = int(np.argmax(ok.sum(axis=1) >= 20)) if (ok.sum(axis=1) >= 20).any() else 0
    sp = int(np.searchsorted(dates, split))
    spans = dict(train=(first, sp - 1), test=(sp, T - 1))
    out = dict(tiers=[], rules=[])
    masks = {}
    for tid, tlabel, lo, hi in TIERS:
        m = ok & (rank >= lo) & (rank <= hi)
        masks[tid] = m
        out["tiers"].append(dict(id=tid, label=tlabel, names={per: float(np.median(m[a : b + 1].sum(axis=1))) for per, (a, b) in spans.items() if b - a > 200}))
    for rid in ids:
        label, kind = meta[rid]
        row = dict(id=rid, label=label, kind=kind, tiers={})
        for tid, m in masks.items():
            X = np.where(m & np.isfinite(RR[rid]), RR[rid], np.nan)
            cnt = np.isfinite(X).sum(axis=1)
            r = np.where(cnt > 0, np.nansum(X, axis=1) / np.maximum(cnt, 1), 0.0)
            row["tiers"][tid] = {per: _curve_stats(np.cumprod(1 + r[a : b + 1])) for per, (a, b) in spans.items() if b - a > 200}
        out["rules"].append(row)
    return out


# ----------------------------------------------------------------------------------------
def round2(df, feats, coins, lad, cfg, S, walk_forward, hac_t, log, split):
    has_ml = S["has_ml"]
    by_id = {sid: (label, sc) for sid, label, sc in S["cands"]}
    main_ids = ["s_rand", "s_vol", "s_quiet", "s_comp", "s_ml_pump", "s_ml_ret", "s_ml_ret_calm", "s_near_high"]
    methods = [(i, by_id[i][0], by_id[i][1]) for i in main_ids if i in by_id]
    out = {}
    if has_ml.sum() < 5000:
        return out
    out["regimes"] = regime_split(df, methods, lad, hac_t, has_ml)
    log("    A 장세별 완료")
    out["calibration"] = dict(
        pump=calibration(df, S["ml_pump"], [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 1.01], hac_t, "pump"),
        ret=calibration(df, S["ml_ret"], [0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 1.01], hac_t, "win5"),
    )
    out["ksweep"] = k_sweep(df, [m for m in methods if m[0] in ("s_rand", "s_ml_pump", "s_ml_ret")], hac_t, has_ml)
    out["liquid"] = liquid_only(df, methods, lad, hac_t, has_ml)
    log("    B·C·G 확신도·종목 수·거래대금 상위 완료")
    need = pd.Series(has_ml, index=df.index)
    out["exits"] = exit_grid(coins, {m[0]: (m[1], picks_of(df, m[2], need)) for m in methods if m[0] in ("s_rand", "s_ml_pump", "s_ml_ret", "s_quiet")}, cfg["cost"])
    log("    D 익절·손절 조합 완료")
    out["long"] = long_horizon(df, feats, lad, walk_forward, hac_t, log)
    log("    E 4주 보유 완료")
    out["majors"] = majors_rules(coins, cfg["cost"], split)
    log("    F 메이저 규칙 완료")
    out["tiers"] = tier_rules(coins, cfg["cost"], split, cfg["min_value"], S["skip"])
    log("    H 거래대금 구간별 규칙 완료")
    return out
