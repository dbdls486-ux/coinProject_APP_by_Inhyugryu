#!/usr/bin/env python3
"""
4차 검증 — (1) 계단식 하락 뒤 거래량을 싣고 상단을 돌파하는 패턴, (2) 크게 오른 경우들의 공통점.

  A. 가설 묶음 3: 계단식 하락 → 거래량 동반 상단 돌파 (하락 깊이·계단 모양·돌파 기준·거래량·당일 봉 크기·BTC 장세를 전부 바꿔 가며)
     가설 묶음 2(횡보 뒤 돌파)도 같은 잣대로 다시 채점해 나란히 놓는다.
  B. 같은 돌파라도 직전 상태(좁은 횡보 / 계단식 하락 / 이미 상승 중 …)에 따라 어떻게 달라지는가
  C. 중앙값 말고 분포: 평균, 상위 10% 수익, 28일 +50% 이상 비율(대박), −20% 이하 비율(실패)
  D. 돌파 신호들 가운데 크게 오른 경우와 나머지를 가르는 진입 직전의 특징 (두 기간에서 같은 방향인 것만)
  E. 시장 전체에서 '28일 안에 2배'가 된 경우의 직전 공통점과, 그 조건으로 샀을 때의 대가

바이낸스를 기준으로 보고, 빗썸은 같은 방향인지 확인하는 용도로 같이 돌린다.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "ml3"), os.path.join(HERE, "..", "ml")):
    sys.path.insert(0, p)
import study as S  # noqa: E402
import study3 as T  # noqa: E402
from feat4 import LABEL4, features4  # noqa: E402

OUT = S.OUT
log = S.log
BIG, LOSS, DOUBLE = 0.5, -0.2, 1.0
ORDER = ["binance", "bithumb"]
PROF = ["vr", "roc1", "day_body", "clv", "range60_prev", "drop30p", "drop60p", "dd120p", "below20_30p", "ll_run_p", "lh_run_p", "vr10p", "vol_trend", "upvol10", "absorb10", "w_pos52", "w_d_sma20", "w_slope20",
        "w_rsi", "w_fib_retr", "w_vol_trend", "rsi14", "d_sma50", "d_sma200", "d_high250", "atr_pct", "val20_log", "price_log", "age", "pumps90", "days_since_pump", "max30", "rs28", "beta60", "corr60", "prem5", "fund7",
        "btc_d_sma50", "btc_roc7", "btc_roc28", "btc_rsi14", "breadth50", "mkt_ret7", "mkt_brk", "newlows10", "dd_gap", "since_brk250"]


def lab(c):
    return LABEL4.get(c) or T.col_label(c)


def add_features4(df, coins):
    bars = df.bar.to_numpy()
    X, cols = None, None
    for sym, idx in df.groupby("sym").indices.items():
        c = coins[sym]
        F = features4(c["o"], c["h"], c["l"], c["c"], c["v"])
        if cols is None:
            cols = list(F.columns)
            X = np.full((len(df), len(cols)), np.nan, dtype="float32")
        X[idx] = F.to_numpy()[bars[idx]]
    df = pd.concat([df, pd.DataFrame(X, columns=cols, index=df.index)], axis=1)
    brk = ((df.brk60.to_numpy() > 0.5) & (df.vr.to_numpy() >= 2)).astype(float)
    df["mkt_brk"] = pd.Series(brk).groupby(df.t.to_numpy()).transform("mean").to_numpy()
    return df


def dist_stats(df, m):
    out = {}
    f7, f28, e28, mx = df.fwd.to_numpy(), df.fwd28.to_numpy(), df.ex28.to_numpy(), df.mx28.to_numpy()
    for per, pm in (("train", ~df.is_test.to_numpy()), ("test", df.is_test.to_numpy())):
        k = m & pm & np.isfinite(f28)
        n = int(k.sum())
        if n < 50:
            out[per] = dict(n=n)
            continue
        x = f28[k]
        out[per] = dict(n=n, net=float(f7[k].mean()), net28=float(x.mean()), med28=float(np.median(x)), p90=float(np.quantile(x, 0.9)), p10=float(np.quantile(x, 0.1)), big=float((x >= BIG).mean()), loss=float((x <= LOSS).mean()),
                        win28=float((x > 0).mean()), ex28=float(np.nanmean(e28[k])), peak=float(np.nanmean(mx[k])))
    return out


def summ(rows, dims):
    full = [r for r in rows if "net28" in r["train"] and "net28" in r["test"]]
    if not full:
        return dict(variants=len(rows), usable=0)
    med = lambda xs: float(np.median(xs)) if len(xs) else None
    keys = ("net", "net28", "med28", "p90", "big", "loss", "win28", "ex28")
    out = dict(variants=len(rows), usable=len(full),
               both_net28_pos=float(np.mean([r["train"]["net28"] > 0 and r["test"]["net28"] > 0 for r in full])), both_med28_pos=float(np.mean([r["train"]["med28"] > 0 and r["test"]["med28"] > 0 for r in full])),
               test_net28_pos=float(np.mean([r["test"]["net28"] > 0 for r in full])),
               median={p: {k: med([r[p][k] for r in full]) for k in keys} for p in ("train", "test")}, byDim={})
    for d in dims:
        vals = {}
        for r in full:
            vals.setdefault(r["p"][d], []).append(r)
        out["byDim"][d] = [dict(value=v, n=len(rs), train=dict(net28=med([r["train"]["net28"] for r in rs]), big=med([r["train"]["big"] for r in rs]), loss=med([r["train"]["loss"] for r in rs])),
                                test=dict(net28=med([r["test"]["net28"] for r in rs]), med28=med([r["test"]["med28"] for r in rs]), big=med([r["test"]["big"] for r in rs]), loss=med([r["test"]["loss"] for r in rs])),
                                both_pos=float(np.mean([r["train"]["net28"] > 0 and r["test"]["net28"] > 0 for r in rs]))) for v, rs in vals.items()]
    out["top"] = sorted(full, key=lambda r: min(r["train"]["net28"], r["test"]["net28"]), reverse=True)[:8]
    return out


def families(df):
    g = lambda k: np.nan_to_num(df[k].to_numpy())
    one = np.ones(len(df), bool)
    btc = df.btc_d_sma50.to_numpy()
    regimes = [("전체", one), ("BTC 50일선 위", btc > 0), ("BTC 50일선 아래", btc <= 0)]
    vr, roc1 = df.vr.to_numpy(), df.roc1.to_numpy()
    d30, d60 = df.drop30p.to_numpy(), df.drop60p.to_numpy()
    out = {}
    # 가설 3: 계단식 하락 → 거래량 동반 상단 돌파
    rows = []
    depth = [("30일간 −15% 이하", d30 <= -0.15), ("30일간 −25% 이하", d30 <= -0.25), ("60일간 −30% 이하", d60 <= -0.30), ("60일간 −50% 이하", d60 <= -0.50)]
    shape = [("조건 없음", one), ("고점·저점을 2번 이상 연속 낮춤", (g("lh_run_p") >= 2) & (g("ll_run_p") >= 2)), ("직전 30일의 70% 이상을 20일선 아래에서 보냄", g("below20_30p") >= 0.7)]
    brk = [("10일 고점", g("brk10") == 1), ("20일 고점", g("brk20") == 1), ("직전 확정 고점", g("brk_piv") == 1), ("50일선 회복", g("reclaim50") == 1)]
    for dn, dm in depth:
        for sn, sm in shape:
            for bn, bm in brk:
                for v in (2, 3, 5):
                    for cn, cm in (("조건 없음", one), ("당일 +8% 이상", roc1 >= 0.08)):
                        for rn, rm in regimes:
                            m = T.dedup(df, dm & sm & bm & (vr >= v) & cm & rm)
                            rows.append(dict(p={"하락 깊이": dn, "계단 모양": sn, "돌파 기준": bn, "거래량 배수": f"{v}배 이상", "당일 봉": cn, "BTC 장세": rn}, **dist_stats(df, m)))
    out["stair"] = summ(rows, ["하락 깊이", "계단 모양", "돌파 기준", "거래량 배수", "당일 봉", "BTC 장세"])
    # 가설 2: 횡보 → 거래량 동반 돌파 (같은 잣대로 다시)
    rows = []
    rng, quiet = df.range60_prev.to_numpy(), df.days_since_pump.to_numpy()
    for rv in (0.3, 0.45, 0.6, 0.9, 99):
        for q in (0, 60, 120, 250):
            for b in ("brk20", "brk60", "brk250"):
                for v in (1.5, 2, 3, 5):
                    for rn, rm in regimes:
                        m = T.dedup(df, (rng < rv) & (quiet > q) & (g(b) == 1) & (vr >= v) & rm)
                        rows.append(dict(p={"직전 60일 범위": "제한 없음" if rv > 9 else f"{round(rv * 100)}% 미만", "급등 없던 기간": "제한 없음" if q == 0 else f"{q}일 넘게", "돌파 기준": {"brk20": "20일 고점", "brk60": "60일 고점", "brk250": "250일 고점"}[b], "거래량 배수": f"{v}배 이상", "BTC 장세": rn}, **dist_stats(df, m)))
    out["base"] = summ(rows, ["직전 60일 범위", "급등 없던 기간", "돌파 기준", "거래량 배수", "BTC 장세"])
    # 같은 돌파(20일 고점 + 거래량 3배), 다른 직전 상태
    trig = (g("brk20") == 1) & (vr >= 3)
    stair = (d30 <= -0.25) & ((g("ll_run_p") >= 2) | (g("below20_30p") >= 0.7))
    states = [("전체", one), ("좁은 횡보 (60일 범위 45% 미만)", rng < 0.45), ("넓은 횡보 (범위 45~90%, 30일 등락 ±15% 이내)", (rng >= 0.45) & (rng < 0.9) & (np.abs(d30) < 0.15)),
              ("계단식 하락 (30일 −25% 이하 + 저점 연속 낮춤 또는 20일선 아래 70%)", stair), ("완만한 하락 (30일 −10~−25%)", (d30 <= -0.10) & (d30 > -0.25)), ("이미 상승 중 (30일 +30% 이상)", d30 >= 0.30)]
    ctx = []
    for sn, sm in states:
        for rn, rm in regimes:
            ctx.append(dict(state=sn, regime=rn, **dist_stats(df, T.dedup(df, trig & sm & rm))))
    out["context"] = ctx
    return out


EVENTS = {
    "base": ("횡보 뒤 거래량 돌파 (60일 범위 60% 미만 + 60일 고점 돌파 + 거래량 2배)", lambda d, g: (d.range60_prev.to_numpy() < 0.6) & (g("brk60") == 1) & (d.vr.to_numpy() >= 2)),
    "stair": ("계단식 하락 뒤 거래량 돌파 (30일 −25% 이하 + 20일 고점 또는 직전 고점 돌파 + 거래량 2배)", lambda d, g: (d.drop30p.to_numpy() <= -0.25) & ((g("brk20") == 1) | (g("brk_piv") == 1)) & (d.vr.to_numpy() >= 2)),
    "any": ("거래량 2배 동반 20일 고점 돌파 전부", lambda d, g: (g("brk20") == 1) & (d.vr.to_numpy() >= 2)),
    # 아래 셋은 위 분석에서 나온 '크게 오른 경우의 공통점'을 조건으로 붙인 것 (BTC RSI 70 은 고르는 기간의 나무가 찾은 기준 70.6 을 반올림)
    "base_hot": ("횡보 뒤 거래량 돌파 + BTC RSI 70 초과 (BTC가 강하게 오르는 중)", lambda d, g: (d.range60_prev.to_numpy() < 0.6) & (g("brk60") == 1) & (d.vr.to_numpy() >= 2) & (d.btc_rsi14.to_numpy() > 70)),
    "base_hot_mod": ("  + 당일 상승 15% 미만 + 최근 28일 BTC 대비 20%p 넘게 앞서지 않음", lambda d, g: (d.range60_prev.to_numpy() < 0.6) & (g("brk60") == 1) & (d.vr.to_numpy() >= 2) & (d.btc_rsi14.to_numpy() > 70) & (d.roc1.to_numpy() < 0.15) & (d.rs28.to_numpy() < 0.2)),
    "base_cold": ("대조: 횡보 뒤 거래량 돌파 + BTC RSI 70 이하", lambda d, g: (d.range60_prev.to_numpy() < 0.6) & (g("brk60") == 1) & (d.vr.to_numpy() >= 2) & (d.btc_rsi14.to_numpy() <= 70)),
    "any_hot": ("거래량 2배 동반 20일 고점 돌파 + BTC RSI 70 초과 + 변동성(ATR) 10% 이하", lambda d, g: (g("brk20") == 1) & (d.vr.to_numpy() >= 2) & (d.btc_rsi14.to_numpy() > 70) & (d.atr_pct.to_numpy() <= 0.10)),
}


def fmt(x):
    return f"{x:.3g}"


def profile(df, m, coins, cost):
    """신호들 가운데 크게 오른 경우(28일 +50% 이상)와 나머지를 가르는 진입 직전의 특징"""
    sub = df[m & df.fwd28.notna().to_numpy()]
    out = dict(stats=T.event_stats(df, m, coins, cost), dist=dist_stats(df, m), winners=[], losers=[], tree=None, examples=[])
    tr, te = sub[~sub.is_test], sub[sub.is_test]
    if len(tr) < 150 or len(te) < 150:
        return out
    for key, name, fn in (("winners", "big", lambda s: s.fwd28 >= BIG), ("losers", "loss", lambda s: s.fwd28 <= LOSS)):
        rows = []
        for c in PROF:
            r = dict(feature=c, label=lab(c))
            okc = True
            for per, s in (("train", tr), ("test", te)):
                x, y = s[c].to_numpy(dtype=float), fn(s).to_numpy()
                k = np.isfinite(x)
                if k.sum() < 100 or y[k].sum() < 15 or (~y[k]).sum() < 15:
                    okc = False
                    break
                r[per] = dict(auc=float(roc_auc_score(y[k], x[k])), hit=float(np.median(x[k][y[k]])), rest=float(np.median(x[k][~y[k]])))
            if okc:
                rows.append(r)
        same = [r for r in rows if (r["train"]["auc"] - 0.5) * (r["test"]["auc"] - 0.5) > 0]
        same.sort(key=lambda r: -min(abs(r["train"]["auc"] - 0.5), abs(r["test"]["auc"] - 0.5)))
        out[key] = same[:12]
    # 조건 조합 (나무): 고르는 기간 신호로 키우고 확인 기간 신호에 적용
    X = lambda s: np.nan_to_num(s[PROF].to_numpy(dtype="float32"), nan=-999.0, posinf=-999.0, neginf=-999.0)
    y = tr.fwd28.clip(-0.5, 1.5).to_numpy()
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(40, int(0.06 * len(tr))), random_state=0).fit(X(tr), y)
    Tt = tree.tree_
    paths = {}

    def walk(node, conds):
        if Tt.children_left[node] == -1:
            paths[node] = conds
            return
        c, thr = PROF[Tt.feature[node]], float(Tt.threshold[node])
        txt = (lambda left: f"{lab(c)}: {'값 없음' if left else '값 있음'}") if thr < -100 else (lambda left: f"{lab(c)} {'≤' if left else '>'} {fmt(thr)}")
        walk(Tt.children_left[node], conds + [txt(True)])
        walk(Tt.children_right[node], conds + [txt(False)])

    walk(0, [])
    ltr, lte = tree.apply(X(tr)), tree.apply(X(te))
    leaves = []
    st = lambda x: dict(n=int(len(x)), net28=float(x.mean()), med28=float(np.median(x)), big=float((x >= BIG).mean()), loss=float((x <= LOSS).mean())) if len(x) >= 20 else dict(n=int(len(x)))
    for node, conds in paths.items():
        leaves.append(dict(rule=conds, train=st(tr.fwd28.to_numpy()[ltr == node]), test=st(te.fwd28.to_numpy()[lte == node])))
    leaves.sort(key=lambda r: -(r["train"].get("net28") or -9))
    out["tree"] = leaves
    top = te.sort_values("fwd28", ascending=False).head(12)
    out["examples"] = [dict(sym=r.sym, date=r.date, net28=float(r.fwd28), peak=float(r.mx28), vr=float(r.vr), roc1=float(r.roc1), drop30=float(r.drop30p), range60=float(r.range60_prev), btc_up=bool(r.btc_d_sma50 > 0), mkt_brk=float(r.mkt_brk)) for r in top.itertuples()]
    return out


def doubles(df):
    """시장 전체에서 '28일 안에 고가 기준 2배'가 된 경우의 직전 조건, 그리고 그 조건으로 샀을 때의 결과"""
    g = lambda k: np.nan_to_num(df[k].to_numpy())
    conds = {sid: (f"{grp} — {label.strip()}", np.nan_to_num(np.asarray(mask, float)).astype(bool)) for sid, (grp, label, mask) in T.setups(df).items() if not sid.startswith("rand") and not sid.endswith("0")}
    wup = (df.w_d_sma20.to_numpy() > 0) & (df.w_slope20.to_numpy() > 0)
    extra = [
        ("동전주 (가격 단위 하위 20%)", df.r_price_log.to_numpy() < 0.2), ("상장 1년 미만", df.age.to_numpy() < 365), ("최근 90일 안에 급등한 적 있음", df.pumps90.to_numpy() > 0), ("변동성 상위 20%", df.r_atr_pct.to_numpy() > 0.8),
        ("변동성 하위 20%", df.r_atr_pct.to_numpy() < 0.2), ("거래대금 하위 30%", df.r_val20_log.to_numpy() < 0.3), ("거래대금 상위 30%", df.r_val20_log.to_numpy() > 0.7), ("52주 범위 하단 25%", df.w_pos52.to_numpy() < 0.25),
        ("52주 범위 상단 25%", df.w_pos52.to_numpy() > 0.75), ("주봉 상승 추세", wup), ("BTC 50일선 위", df.btc_d_sma50.to_numpy() > 0), ("BTC 50일선 아래", df.btc_d_sma50.to_numpy() <= 0),
        ("그날 돌파 종목이 전체의 5% 이상", df.mkt_brk.to_numpy() >= 0.05), ("당일 거래량 3배 이상", df.vr.to_numpy() >= 3), ("당일 +10% 이상", df.roc1.to_numpy() >= 0.10), ("250일 고점 돌파", g("brk250") == 1),
        ("펀딩비 음수", df.fund7.to_numpy() < 0), ("30일간 −25% 이하로 빠진 상태", df.drop30p.to_numpy() <= -0.25), ("직전 60일 범위 45% 미만 (좁은 횡보)", df.range60_prev.to_numpy() < 0.45),
    ]
    for k, (lab_, fn) in EVENTS.items():
        extra.append((lab_, fn(df, g)))
    for i, (lab_, m) in enumerate(extra):
        conds[f"x{i}"] = (lab_, np.nan_to_num(np.asarray(m, float)).astype(bool))
    mx, f28 = df.mx28.to_numpy(), df.fwd28.to_numpy()
    ok = np.isfinite(mx) & np.isfinite(f28)
    base = {}
    rows = []
    for per, pm in (("train", ~df.is_test.to_numpy()), ("test", df.is_test.to_numpy())):
        k = ok & pm
        base[per] = dict(rate=float((mx[k] >= DOUBLE).mean()), net28=float(f28[k].mean()), loss=float((f28[k] <= -0.3).mean()), n=int(k.sum()))
    for cid, (label, m) in conds.items():
        r = dict(label=label)
        for per, pm in (("train", ~df.is_test.to_numpy()), ("test", df.is_test.to_numpy())):
            k = ok & pm & m
            if k.sum() < 100:
                r[per] = dict(n=int(k.sum()))
                continue
            rate = float((mx[k] >= DOUBLE).mean())
            r[per] = dict(n=int(k.sum()), rate=rate, lift=rate / base[per]["rate"] if base[per]["rate"] else None, cover=float(((mx >= DOUBLE) & k).sum() / max(((mx >= DOUBLE) & ok & pm).sum(), 1)), net28=float(f28[k].mean()), med28=float(np.median(f28[k])), loss=float((f28[k] <= -0.3).mean()))
        rows.append(r)
    full = [r for r in rows if "lift" in r["train"] and "lift" in r["test"]]
    full.sort(key=lambda r: -min(r["train"]["lift"], r["test"]["lift"]))
    return dict(base=base, rows=full)


# ----------------------------------------------------------------------------------------
def pc(x, d=1):
    return "-" if x is None else f"{x * 100:+.{d}f}%"


def sh(x):
    return "-" if x is None else f"{x * 100:.0f}%"


def render(R):
    L = []
    A = L.append
    A("# 계단식 하락 뒤 돌파 · 크게 오른 경우의 공통점")
    A("")
    A(f"생성 {R['generatedAt'].replace('T', ' ')} UTC · 신호는 종가에, 매수는 다음 날 시가에 · 수수료·체결 오차 반영 · 고르는 기간 ~2023 / 확인 기간 2024~ · 수익은 28일 보유 기준")
    A("")
    A("**대박** = 28일 뒤 +50% 이상, **실패** = 28일 뒤 −20% 이하, **상위 10%** = 그 신호들 가운데 수익 상위 10% 경계값.")
    A("")
    for m in R["markets"]:
        A(f"## {m['label']} ({m['coins']}종목)")
        A("")
        for key, title in (("stair", "A. 가설 묶음 3 — 계단식 하락 뒤 거래량을 싣고 상단 돌파"), ("base", "A'. 가설 묶음 2 — 횡보 뒤 거래량 동반 돌파 (같은 잣대)")):
            F = m["families"][key]
            if not F.get("usable"):
                continue
            a, b = F["median"]["train"], F["median"]["test"]
            A(f"### {title}")
            A(f"조합 {F['variants']}개 중 두 기간 모두 50건 이상인 {F['usable']}개 기준. 두 기간 모두 28일 평균이 플러스인 조합 {sh(F['both_net28_pos'])}, 두 기간 모두 28일 중앙값이 플러스인 조합 {sh(F['both_med28_pos'])}.")
            A("")
            A("| 조합들의 중앙값 | 28일 평균 | 28일 중앙값 | 상위 10% | 대박 비율 | 실패 비율 | 오른 비율 | 또래 대비 |")
            A("|---|---|---|---|---|---|---|---|")
            A(f"| 고르는 기간 | {pc(a['net28'])} | {pc(a['med28'])} | {pc(a['p90'])} | {sh(a['big'])} | {sh(a['loss'])} | {sh(a['win28'])} | {pc(a['ex28'])} |")
            A(f"| 확인 기간 | {pc(b['net28'])} | {pc(b['med28'])} | {pc(b['p90'])} | {sh(b['big'])} | {sh(b['loss'])} | {sh(b['win28'])} | {pc(b['ex28'])} |")
            A("")
            A("| 조건 | 값 | 조합 수 | 28일 평균 (고름/확인) | 확인 중앙값 | 대박 비율 (고름/확인) | 실패 비율 (고름/확인) | 두 기간 모두 플러스 |")
            A("|---|---|---|---|---|---|---|---|")
            for d, vals in F["byDim"].items():
                for v in vals:
                    A(f"| {d} | {v['value']} | {v['n']} | {pc(v['train']['net28'])} / {pc(v['test']['net28'])} | {pc(v['test']['med28'])} | {sh(v['train']['big'])} / {sh(v['test']['big'])} | {sh(v['train']['loss'])} / {sh(v['test']['loss'])} | {sh(v['both_pos'])} |")
            A("")
            A("두 기간 모두 성적이 좋았던 조합 8개 (결과를 보고 고른 것이라 공통 조건을 보는 용도)")
            A("")
            A("| 조합 | 건수 | 28일 평균 | 중앙값 | 대박 비율 | 실패 비율 |")
            A("|---|---|---|---|---|---|")
            for r in F["top"]:
                a, b = r["train"], r["test"]
                A(f"| {' · '.join(f'{k} {v}' for k, v in r['p'].items())} | {a['n']} / {b['n']} | {pc(a['net28'])} / {pc(b['net28'])} | {pc(a['med28'])} / {pc(b['med28'])} | {sh(a['big'])} / {sh(b['big'])} | {sh(a['loss'])} / {sh(b['loss'])} |")
            A("")
        A("### B. 같은 돌파(20일 고점 + 거래량 3배), 직전 상태에 따라")
        A("| 직전 상태 | BTC 장세 | 건수 (고름/확인) | 28일 평균 | 중앙값 | 상위 10% | 대박 비율 | 실패 비율 | 28일 안 최고점 평균 |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in m["families"]["context"]:
            a, b = r["train"], r["test"]
            if "net28" not in a or "net28" not in b:
                A(f"| {r['state']} | {r['regime']} | {a.get('n', 0)} / {b.get('n', 0)} | - | - | - | - | - | - |")
                continue
            A(f"| {r['state']} | {r['regime']} | {a['n']} / {b['n']} | {pc(a['net28'])} / {pc(b['net28'])} | {pc(a['med28'])} / {pc(b['med28'])} | {pc(a['p90'])} / {pc(b['p90'])} | {sh(a['big'])} / {sh(b['big'])} | {sh(a['loss'])} / {sh(b['loss'])} | {pc(a['peak'])} / {pc(b['peak'])} |")
        A("")
        if m.get("btcHot"):
            h = m["btcHot"]
            A(f"BTC RSI가 70을 넘은 날: 고르는 기간 {h['train']['days']}일 ({sh(h['train']['share'])}, 묶음 {h['train']['episodes']}개), 확인 기간 {h['test']['days']}일 ({sh(h['test']['share'])}, 묶음 {h['test']['episodes']}개). 묶음 = 10일 넘게 떨어지면 따로 센 것.")
            A("")
        for key, (title, _) in EVENTS.items():
            P = m["profiles"].get(key)
            if not P:
                continue
            A(f"### D. 크게 오른 경우의 공통점 — {title.strip()}")
            d = P["dist"]
            if "net28" in d.get("train", {}) and "net28" in d.get("test", {}):
                a, b = d["train"], d["test"]
                A(f"건수 {a['n']} / {b['n']} · 28일 평균 {pc(a['net28'])} / {pc(b['net28'])} · 중앙값 {pc(a['med28'])} / {pc(b['med28'])} · 대박 비율 {sh(a['big'])} / {sh(b['big'])} · 실패 비율 {sh(a['loss'])} / {sh(b['loss'])}")
                A("")
            ex = P["stats"]
            acc = [((ex[p].get("exits") or {}).get("account")) for p in ("train", "test")]
            if all(acc):
                A("손절(최근 10일 저점)·위험의 2배 익절로 10칸 계좌를 굴리면: " + " / ".join(f"연 {pc(a['cagr'], 0)}, 최대 낙폭 {pc(a['mdd'], 0)}, 진입 {a['taken']}건" for a in acc) + " (고르는 기간 / 확인 기간)")
                A("")
            if ex.get("yearly"):
                A("연도별 (건수 · 7일 · 28일): " + " / ".join(f"{y} {v['n']}건 {pc(v['net'])} {pc(v['net28'])}" for y, v in ex["yearly"].items()))
                A("")
            for k2, t2 in (("winners", "대박(28일 +50% 이상)이 된 신호와 나머지를 가르는 특징"), ("losers", "실패(28일 −20% 이하)한 신호와 나머지를 가르는 특징")):
                if not P[k2]:
                    continue
                A(f"**{t2}** — 두 기간에서 같은 방향인 것만, 가르는 힘이 큰 순. 가르는 힘 50%는 구분 못 함, 60% 이상이면 뚜렷.")
                A("")
                A("| 특징 | 해당 신호의 중앙값 (고름/확인) | 나머지의 중앙값 (고름/확인) | 가르는 힘 (고름/확인) |")
                A("|---|---|---|---|")
                for r in P[k2]:
                    a, b = r["train"], r["test"]
                    A(f"| {r['label']} | {fmt(a['hit'])} / {fmt(b['hit'])} | {fmt(a['rest'])} / {fmt(b['rest'])} | {sh(max(a['auc'], 1 - a['auc']))} / {sh(max(b['auc'], 1 - b['auc']))} ({'클수록' if a['auc'] > 0.5 else '작을수록'}) |")
                A("")
            if P["tree"]:
                A("**조건 조합** (고르는 기간의 신호로 찾고 확인 기간에 적용)")
                A("")
                A("| 조건 조합 | 고르는 기간: 건수 · 28일 평균 · 중앙값 · 대박 | 확인 기간: 건수 · 28일 평균 · 중앙값 · 대박 · 실패 |")
                A("|---|---|---|")
                for r in P["tree"]:
                    a, b = r["train"], r["test"]
                    ta = f"{a['n']} · {pc(a['net28'])} · {pc(a['med28'])} · {sh(a['big'])}" if "net28" in a else f"{a['n']}건"
                    tb = f"{b['n']} · {pc(b['net28'])} · {pc(b['med28'])} · {sh(b['big'])} · {sh(b['loss'])}" if "net28" in b else f"{b['n']}건"
                    A(f"| {' → '.join(r['rule'])} | {ta} | {tb} |")
                A("")
            if P["examples"]:
                A("**확인 기간에 가장 많이 오른 신호 12개** (차트에서 직접 확인용)")
                A("")
                A("| 종목 | 신호일 | 28일 수익 | 28일 안 최고점 | 당일 거래량 배수 | 당일 등락 | 직전 30일 등락 | 직전 60일 범위 | BTC 50일선 위 | 그날 돌파 종목 비율 |")
                A("|---|---|---|---|---|---|---|---|---|---|")
                for e in P["examples"]:
                    A(f"| {e['sym']} | {e['date']} | {pc(e['net28'], 0)} | {pc(e['peak'], 0)} | {e['vr']:.1f}배 | {pc(e['roc1'])} | {pc(e['drop30'], 0)} | {sh(e['range60'])} | {'예' if e['btc_up'] else '아니오'} | {e['mkt_brk'] * 100:.1f}% |")
                A("")
        D = m["doubles"]
        A("### E. 28일 안에 2배가 된 경우의 직전 조건")
        A(f"아무 코인·아무 날에 샀을 때 28일 안에 고가 기준 2배가 될 확률: 고르는 기간 {D['base']['train']['rate'] * 100:.1f}%, 확인 기간 {D['base']['test']['rate'] * 100:.1f}%. 28일 평균 수익 {pc(D['base']['train']['net28'])} / {pc(D['base']['test']['net28'])}, −30% 이하로 끝난 비율 {sh(D['base']['train']['loss'])} / {sh(D['base']['test']['loss'])}.")
        A("")
        A("| 직전 조건 | 건수 (고름/확인) | 2배 확률 | 평소 대비 배수 | 전체 2배 사례 중 차지하는 비율 (확인) | 그 조건으로 샀을 때 28일 평균 | 중앙값 | −30% 이하 비율 |")
        A("|---|---|---|---|---|---|---|---|")
        for r in D["rows"]:
            a, b = r["train"], r["test"]
            A(f"| {r['label']} | {a['n']} / {b['n']} | {a['rate'] * 100:.1f}% / {b['rate'] * 100:.1f}% | {a['lift']:.1f} / {b['lift']:.1f} | {sh(b['cover'])} | {pc(a['net28'])} / {pc(b['net28'])} | {pc(a['med28'])} / {pc(b['med28'])} | {sh(a['loss'])} / {sh(b['loss'])} |")
        A("")
    A("## 읽을 때 주의")
    A("- 28일 수익은 28일째 종가에 팔았을 때다. '28일 안 최고점'과 '2배 확률'은 장중 고가 기준이라 그 가격에 팔 수 있었다는 뜻이 아니다.")
    A("- 크게 오른 경우의 특징은 결과를 알고 나서 나눈 것이다. 두 기간에서 같은 방향인 것만 남겼지만, 그 특징으로 미리 골랐을 때의 성적은 '조건 조합' 표의 확인 기간 칸을 봐야 한다.")
    A("- 조합 수백 개를 한꺼번에 봤다. 한두 조합의 숫자보다 조건별 경향(어느 값에서 좋아지는가)을 본다.")
    return "\n".join(L) + "\n"


def main():
    t0 = time.time()
    raw = {m: S.load_market(m, cfg) for m, cfg in S.MARKETS.items() if os.path.isdir(f"{S.DATA}/{m}")}
    result = dict(generatedAt=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M"), markets=[])
    for mkt in ORDER:
        cfg = S.MARKETS[mkt]
        if mkt not in raw or "BTC" not in raw[mkt]:
            continue
        raw.setdefault("binance" if mkt == "bithumb" else "bithumb", {})
        log(f"[{cfg['label']}] 지표 계산")
        df, feats, all_dates, _ = S.build(mkt, cfg, raw)
        df, _ = T.add_features3(df, raw[mkt])
        df = add_features4(df, raw[mkt])
        log(f"  표 {len(df):,}행")
        fam = families(df)
        log("  A·B 가설 묶음 채점 완료")
        g = lambda k: np.nan_to_num(df[k].to_numpy())
        prof = {k: profile(df, T.dedup(df, np.nan_to_num(np.asarray(fn(df, g), float)).astype(bool)), raw[mkt], cfg["cost"]) for k, (_, fn) in EVENTS.items()}
        log("  D 크게 오른 경우의 공통점 완료")
        dd = df.groupby("t").agg(r=("btc_rsi14", "first"), test=("is_test", "first")).reset_index()
        hot = {}
        for per, pm in (("train", ~dd.test.to_numpy()), ("test", dd.test.to_numpy())):
            tt = dd.t.to_numpy()[pm & (dd.r.to_numpy() > 70)]
            hot[per] = dict(days=int(len(tt)), episodes=int(1 + (np.diff(tt) > 10).sum()) if len(tt) else 0, share=float(len(tt) / max(pm.sum(), 1)))
        dbl = doubles(df)
        log("  E 2배 사례 완료")
        result["markets"].append(dict(market=mkt, label=cfg["label"], rows=int(len(df)), coins=int(df.sym.nunique()), families=fam, profiles=prof, doubles=dbl, btcHot=hot))
        del df

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else round(float(o), 5)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    result = clean(result)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/winners.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(f"{OUT}/WINNERS.md", "w", encoding="utf-8") as f:
        f.write(render(result))
    log(f"끝 ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
