#!/usr/bin/env python3
"""
3차 검증 — 여러 조건을 '단계'로 묶고, 일봉과 주봉을 같이 보고, 뉴스까지 겹친다.

  A. 단계형 시나리오: 준비 조건(주봉 추세·바닥권·투매·횡보 등) → 방아쇠(반등 봉·돌파·거래량) → 확인(저점 지킴·구조 전환)
     시나리오마다 주봉 필터·BTC 장세를 붙인 변형을 같이 본다. 대조군(조건 하나를 뺀 것, 무작위)도 같이 채점한다.
     고정 보유(7일·28일)와 구조 기반 매매(최근 10일 저점 아래 손절, 위험의 2배에서 익절 / 추세 따라가기) 모두 계산한다.
  B. 조건 조합 발굴: 2023년까지의 자료로 얕은 의사결정나무를 키워 "조건 3~4개 조합"을 뽑고, 2024년 이후에 그대로 적용해 본다.
  C. 모델 재학습: 1차의 일봉 지표 103개에 주봉·순서형 지표를 더해 다시 학습, 무엇이 달라지는지 본다.
  D. 뉴스 × 차트: 뉴스가 나온 날을 종류별로 잡고, 그 직전 차트 상태에 따라 이후 수익이 어떻게 갈리는지 본다.

고르는 기간 ~2023-12-31 / 확인 기간 2024-01-01~. 신호는 종가에, 매수는 다음 날 시가에. 수수료·체결 오차 반영.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ml"))
import study as S  # noqa: E402
from feat3 import GRID1, LABEL3, MODEL3, features3  # noqa: E402
from features import BINARY, LABEL  # noqa: E402
from round2 import picks_of, trade_stats  # noqa: E402

FLAGS3 = {"w_align", "w_gc", "w_rsi_turn", "w_macd_up", "w_macd_cross", "w_hl2", "w_ll3", "w_fib_zone", "clx", "clx_any", "retest", "hl", "hh", "struct_turn", "turn10", "brk_piv", "rev_bar", "brk10", "rsi_up40", "near_sma20", "touch50_5", "sq5", "btc_dd_recent", "btc_reclaim"}
OUT = S.OUT
NEWS = os.environ.get("LAB_NEWS", "lab/.news")
RNG = np.random.default_rng(31)
GAP = 5  # 같은 코인·같은 시나리오 신호는 5일 안에 다시 세지 않는다
log = S.log
hac_t = S.hac_t


# ----------------------------------------------------------------------------------------
def add_features3(df, coins):
    btc = pd.Series(coins["BTC"]["c"], index=coins["BTC"]["dates"])
    cols = None
    X = None
    bars = df.bar.to_numpy()
    for sym, idx in df.groupby("sym").indices.items():
        c = coins[sym]
        F = features3(c["o"], c["h"], c["l"], c["c"], c["v"], btc.reindex(c["dates"]).ffill().to_numpy())
        if cols is None:
            cols = list(F.columns)
            X = np.full((len(df), len(cols)), np.nan, dtype="float32")
        X[idx] = F.to_numpy()[bars[idx]]
    return pd.concat([df, pd.DataFrame(X, columns=cols, index=df.index)], axis=1), cols


def dedup(df, mask, gap=GAP):
    idx = np.flatnonzero(mask)
    sym, bar = df.sym.to_numpy()[idx], df.bar.to_numpy()[idx]
    keep = np.ones(len(idx), bool)
    last_sym, last_bar = None, -(10**9)
    for k in range(len(idx)):
        if sym[k] == last_sym and bar[k] - last_bar <= gap:
            keep[k] = False
        else:
            last_sym, last_bar = sym[k], bar[k]
    out = np.zeros(len(df), bool)
    out[idx[keep]] = True
    return out


def setups(df):
    g = lambda k: df[k].to_numpy()
    wup = (g("w_d_sma20") > 0) & (g("w_slope20") > 0)
    wbase = g("w_pos52") < 0.25
    btc_up = g("btc_d_sma50") > 0
    rev = g("rev_bar") == 1
    vr = g("vr")
    dsc, hold = g("dsc"), g("hold")
    D = {}
    D["s1a"] = ("1. 투매 뒤 버팀", "① 투매 당일: 저점 갱신이 이어지다 거래량 3배 + 신저가인데 종가는 회복", g("clx") == 1)
    D["s1a0"] = ("1. 투매 뒤 버팀", "대조: 거래량 3배 + 신저가 (종가 회복 조건 없음)", g("clx_any") == 1)
    D["s1b"] = ("1. 투매 뒤 버팀", "② 투매 뒤 3일간 저점을 지킴", (dsc == 3) & (hold >= 0))
    D["s1b0"] = ("1. 투매 뒤 버팀", "대조: 투매 뒤 3일 안에 저점이 깨짐", (dsc == 3) & (hold < 0))
    s1c = (dsc >= 3) & (dsc <= 10) & (hold >= 0) & rev & (g("rsc") > 0)
    D["s1c"] = ("1. 투매 뒤 버팀", "③ 3~10일 저점 지키고, 전일 고가를 넘는 양봉", s1c)
    D["s1d"] = ("1. 투매 뒤 버팀", "③ + 52주 범위 하단 25% (바닥권)", s1c & wbase)
    D["s1e"] = ("1. 투매 뒤 버팀", "③ + BTC 50일선 위", s1c & btc_up)
    D["s1f"] = ("1. 투매 뒤 버팀", "③ + 투매 이후에도 거래량 유지 (평균 1.5배 이상)", s1c & (g("vol_after") >= 1.5))
    s2 = (g("retest") == 1) & rev
    D["s2a"] = ("2. 저점 재확인 (쌍바닥)", "투매 저점 근처로 다시 내려왔다가 거래량이 줄며 양봉", s2)
    D["s2b"] = ("2. 저점 재확인 (쌍바닥)", "  + 주봉 RSI 과매도 뒤 반등", s2 & (g("w_rsi_turn") == 1))
    s3a = wup & (g("rsi_min5") < 40) & rev
    D["s3a"] = ("3. 주봉 상승 추세 속 일봉 눌림", "주봉 20주선 위·상승 + 일봉 RSI 40 아래까지 눌림 + 전일 고가 넘는 양봉", s3a)
    D["s3b"] = ("3. 주봉 상승 추세 속 일봉 눌림", "  + 거래량 1.5배", s3a & (vr >= 1.5))
    D["s3c"] = ("3. 주봉 상승 추세 속 일봉 눌림", "  + 주봉 피보나치 0.382~0.618 구간", s3a & (g("w_fib_zone") == 1))
    D["s3d"] = ("3. 주봉 상승 추세 속 일봉 눌림", "  + BTC 50일선 위", s3a & btc_up)
    D["s3e"] = ("3. 주봉 상승 추세 속 일봉 눌림", "주봉 정배열 + 50일선까지 눌림 + 양봉", (g("w_align") == 1) & (g("touch50_5") == 1) & rev & (g("d_sma50") > -0.03))
    s4a = (g("w_fib_zone") == 1) & (g("turn10") == 1) & (g("brk_piv") == 1)
    D["s4a"] = ("4. 주봉 피보나치 + 일봉 구조 전환", "주봉 피보나치 0.382~0.618 + 일봉 저점 높임 확정 뒤 직전 고점 돌파", s4a)
    D["s4b"] = ("4. 주봉 피보나치 + 일봉 구조 전환", "  + 주봉 20주선 상승", s4a & wup)
    s5a = (g("sq5") == 1) & (g("brk20") == 1) & (vr >= 2)
    D["s5a"] = ("5. 변동성 수축 뒤 돌파", "볼린저 밴드폭 수축 뒤 20일 고점 돌파 + 거래량 2배", s5a)
    D["s5b"] = ("5. 변동성 수축 뒤 돌파", "  + 주봉 상승 추세", s5a & wup)
    D["s5c"] = ("5. 변동성 수축 뒤 돌파", "  + 주봉 상승 추세 + BTC 50일선 위", s5a & wup & btc_up)
    s6a = (g("range60_prev") < 0.45) & (g("days_since_pump") > 120) & (g("brk60") == 1) & (vr >= 3)
    D["s6a"] = ("6. 긴 횡보 뒤 거래량 돌파", "60일 범위 45% 미만·120일간 급등 없음 → 60일 고점 돌파 + 거래량 3배", s6a)
    D["s6b"] = ("6. 긴 횡보 뒤 거래량 돌파", "  + 거래량이 미리 늘고 있었음 (20일/120일 1.2배 이상)", s6a & (g("vol_trend") >= 1.2))
    D["s7"] = ("7. 주봉 골든크로스 뒤 눌림", "주봉 10·40주 골든크로스 8주 이내 + 20일선 눌림 + 양봉", (g("w_gc") == 1) & (g("near_sma20") == 1) & rev)
    s8a = (g("w_rsi_turn") == 1) & (g("hl") == 1) & (g("d_sma20") > 0)
    D["s8a"] = ("8. 주봉 과매도 반전", "주봉 RSI 과매도 뒤 반등 + 일봉 저점 높임 + 20일선 위", s8a)
    D["s8b"] = ("8. 주봉 과매도 반전", "  + 직전 고점 돌파", s8a & (g("brk_piv") == 1))
    D["s9"] = ("9. 다이버전스 + 거래량", "RSI 상승 다이버전스 + 52주 하단 30% + 최근 5일 거래량 2배 + 양봉", (g("bull_div") == 1) & (g("w_pos52") < 0.3) & (g("vr_max5") >= 2) & rev)
    s10a = (g("since_brk250") >= 5) & (g("since_brk250") <= 30) & (g("near_sma20") == 1) & (g("d_sma50") > 0) & rev
    D["s10a"] = ("10. 신고가 뒤 첫 눌림", "250일 고점 돌파 5~30일 뒤 20일선 눌림 + 양봉", s10a)
    D["s10b"] = ("10. 신고가 뒤 첫 눌림", "  + 주봉 정배열", s10a & (g("w_align") == 1))
    s11a = (g("btc_dd_recent") == 1) & (g("dd_gap") > 0.05) & (g("btc_reclaim") == 1)
    D["s11a"] = ("11. BTC 조정 때 버틴 알트", "BTC가 14일 −10% 조정하는 동안 덜 빠졌고, BTC가 20일선을 회복", s11a)
    D["s11b"] = ("11. BTC 조정 때 버틴 알트", "  + 주봉 상승 추세", s11a & wup)
    D["s12"] = ("12. 매집 흔적 뒤 돌파", "거래량 2배인데 가격 제자리인 날 3일 이상 (10일 중) → 10일 고점 돌파", (g("absorb10") >= 3) & (g("pos60_prev") < 0.5) & (g("brk10") == 1))
    s13a = (g("turn10") == 1) & (g("brk_piv") == 1) & (vr >= 1.5)
    D["s13a"] = ("13. 하락 구조 전환", "저점 낮추다 처음 높임 확정 → 직전 고점 돌파 + 거래량 1.5배", s13a)
    D["s13b"] = ("13. 하락 구조 전환", "  + 52주 범위 하단 40%", s13a & (g("w_pos52") < 0.4))
    D["s13c"] = ("13. 하락 구조 전환", "  + BTC 50일선 위", s13a & btc_up)
    for i in range(1, 4):
        D[f"rand{i}"] = ("대조: 무작위", f"무작위로 뽑은 날 {i}", RNG.random(len(df)) < 4000 / len(df))
    return D


def struct_exits(coins, sub, cost, cap=20000):
    """
    구조 기반 매매. 손절 = 최근 10일 저가의 1% 아래 (위험은 매수가의 4~25%로 제한).
      2R: 위험의 2배에서 익절, 최대 28일 보유
      추세: 익절 없이, 종가가 직전 10일 저가를 깨면 다음 날 시가에 매도, 최대 60일 보유
    """
    if len(sub) > cap:
        sub = sub.sample(cap, random_state=1)
    r2, rr, rt, trades = [], [], [], []
    for sym, bar, sref, t0 in zip(sub.sym.to_numpy(), sub.bar.to_numpy(), sub.stop_ref.to_numpy(), sub.t.to_numpy()):
        c = coins[sym]
        o, h, l, cl = c["o"], c["h"], c["l"], c["c"]
        n, i = len(cl), int(bar)
        if i + 1 >= n:
            continue
        entry = o[i + 1]
        risk = (entry - sref * 0.99) / entry if np.isfinite(sref) else 0.08
        risk = min(max(risk, 0.04), 0.25)
        stop, tgt = entry * (1 - risk), entry * (1 + 2 * risk)
        px = None
        end = min(i + 28, n - 1)
        dx = end
        for d in range(i + 1, end + 1):
            if l[d] <= stop:
                px, dx = min(o[d], stop), d
                break
            if h[d] >= tgt:
                px, dx = max(o[d], tgt), d
                break
        if px is None and i + 28 <= n - 1:
            px = cl[end]
        if px is not None:
            ret = px / entry * (1 - cost) / (1 + cost) - 1
            r2.append(ret)
            rr.append((px / entry - 1) / risk)
            trades.append((int(t0) + 1, int(t0) + (dx - i), ret))
        px = None
        end = min(i + 60, n - 1)
        sell = False
        for d in range(i + 1, end + 1):
            if sell:
                px = o[d]
                break
            if l[d] <= stop:
                px = min(o[d], stop)
                break
            if d - 10 >= 0 and cl[d] < l[d - 10 : d].min():
                sell = True
        if px is None and i + 60 <= n - 1:
            px = cl[end]
        if px is not None:
            rt.append(px / entry * (1 - cost) / (1 + cost) - 1)
    f = lambda x: None if len(x) < 30 else dict(n=len(x), mean=float(np.mean(x)), median=float(np.median(x)), win=float(np.mean(np.array(x) > 0)))
    out = dict(r2=f(r2), trail=f(rt))
    if out["r2"]:
        out["r2"]["R"] = float(np.mean(rr))
        out["account"] = account(trades)
    return out


def account(trades, slots=10):
    """손절·2배 익절 매매를 실제 계좌처럼: 자금을 10칸으로 나눠 신호마다 한 칸씩, 칸이 다 차면 그 신호는 건너뛴다. (청산 시점 기준 손익)"""
    if len(trades) < 30:
        return None
    trades = sorted(trades)
    eq, peak, mdd, skipped = 1.0, 1.0, 0.0, 0
    open_pos = []  # (청산일, 투입액, 수익률)
    for ent, ex, ret in trades:
        still = []
        for p in sorted(open_pos):
            if p[0] <= ent:
                eq += p[1] * p[2]
                peak = max(peak, eq)
                mdd = min(mdd, eq / peak - 1)
            else:
                still.append(p)
        open_pos = still
        if len(open_pos) >= slots:
            skipped += 1
            continue
        open_pos.append((ex, eq / slots, ret))
    for p in sorted(open_pos):
        eq += p[1] * p[2]
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    days = max(trades[-1][1] - trades[0][0], 30)
    return dict(total=float(eq - 1), cagr=float(eq ** (365 / days) - 1) if eq > 0 else -1.0, mdd=float(mdd), taken=len(trades) - skipped, skipped=skipped, years=round(days / 365, 2))


def event_stats(df, mask, coins, cost, with_exits=True):
    out = {}
    for per, pm in (("train", ~df.is_test.to_numpy()), ("test", df.is_test.to_numpy())):
        m = mask & pm
        n = int(m.sum())
        if n < 30:
            out[per] = dict(n=n)
            continue
        sub = df.loc[m, ["t", "sym", "bar", "ex", "fwd", "pump", "fwd28", "ex28", "stop_ref"]]
        base = float(df.pump.to_numpy()[pm].mean())
        d7 = sub.groupby("t").ex.mean().to_numpy()
        s28 = sub[sub.fwd28.notna()]
        d28 = s28.groupby("t").ex28.mean().to_numpy()
        r = dict(
            n=n, days=int(sub.t.nunique()), coins=int(sub.sym.nunique()), pump=float(sub.pump.mean()), lift=float(sub.pump.mean() / base) if base else None,
            net=float(sub.fwd.mean()), median=float(sub.fwd.median()), win=float((sub.fwd > 0).mean()), ex=float(d7.mean()), t=hac_t(d7) if len(d7) >= 60 else None,
        )
        if len(s28) >= 30:
            r.update(net28=float(s28.fwd28.mean()), median28=float(s28.fwd28.median()), win28=float((s28.fwd28 > 0).mean()), ex28=float(d28.mean()), t28=hac_t(d28, 28) if len(d28) >= 60 else None)
        if with_exits:
            r["exits"] = struct_exits(coins, sub, cost)
        out[per] = r
    if with_exits and mask.sum() >= 60:
        yr = df.date.to_numpy()[mask].astype("U4")
        f7, f28 = df.fwd.to_numpy()[mask], df.fwd28.to_numpy()[mask]
        out["yearly"] = {y: dict(n=int((yr == y).sum()), net=float(f7[yr == y].mean()), net28=float(np.nanmean(f28[yr == y])) if np.isfinite(f28[yr == y]).any() else None) for y in sorted(set(yr))}
    return out


def light_stats(df, m):
    """가설 묶음용 가벼운 채점"""
    out = {}
    t, ex, fwd, f28, e28 = df.t.to_numpy(), df.ex.to_numpy(), df.fwd.to_numpy(), df.fwd28.to_numpy(), df.ex28.to_numpy()
    for per, pm in (("train", ~df.is_test.to_numpy()), ("test", df.is_test.to_numpy())):
        k = m & pm
        n = int(k.sum())
        if n < 50:
            out[per] = dict(n=n)
            continue
        d = pd.Series(ex[k]).groupby(t[k]).mean().to_numpy()
        ok28 = np.isfinite(f28[k])
        out[per] = dict(n=n, net=float(fwd[k].mean()), win=float((fwd[k] > 0).mean()), ex=float(d.mean()), t=hac_t(d) if len(d) >= 60 else None,
                        net28=float(f28[k][ok28].mean()) if ok28.sum() >= 30 else None, win28=float((f28[k][ok28] > 0).mean()) if ok28.sum() >= 30 else None, ex28=float(np.nanmean(e28[k])) if ok28.sum() >= 30 else None)
    return out


def summarize_family(rows, dims):
    full = [r for r in rows if "net" in r["train"] and "net" in r["test"]]
    if not full:
        return dict(variants=len(rows), usable=0)
    share = lambda fn: float(np.mean([bool(fn(r)) for r in full]))
    med = lambda xs: float(np.median(xs)) if len(xs) else None
    g = lambda r, p, k: r[p].get(k)
    out = dict(
        variants=len(rows), usable=len(full),
        test_net_pos=share(lambda r: r["test"]["net"] > 0), both_net_pos=share(lambda r: r["train"]["net"] > 0 and r["test"]["net"] > 0),
        both_net28_pos=share(lambda r: (g(r, "train", "net28") or -1) > 0 and (g(r, "test", "net28") or -1) > 0),
        both_ex_sig=share(lambda r: (g(r, "train", "t") or 0) >= 2 and (g(r, "test", "t") or 0) >= 2), both_ex_neg=share(lambda r: (g(r, "train", "t") or 0) <= -2 and (g(r, "test", "t") or 0) <= -2),
        median=dict(train_net=med([r["train"]["net"] for r in full]), test_net=med([r["test"]["net"] for r in full]), train_net28=med([g(r, "train", "net28") for r in full if g(r, "train", "net28") is not None]),
                    test_net28=med([g(r, "test", "net28") for r in full if g(r, "test", "net28") is not None]), train_ex=med([r["train"]["ex"] for r in full]), test_ex=med([r["test"]["ex"] for r in full]),
                    train_win=med([r["train"]["win"] for r in full]), test_win=med([r["test"]["win"] for r in full])),
        byDim={},
    )
    for dname, _ in dims:
        vals = {}
        for r in full:
            vals.setdefault(r["p"][dname], []).append(r)
        out["byDim"][dname] = [dict(value=v, n=len(rs), train_net=med([r["train"]["net"] for r in rs]), test_net=med([r["test"]["net"] for r in rs]),
                                    train_net28=med([g(r, "train", "net28") for r in rs if g(r, "train", "net28") is not None]), test_net28=med([g(r, "test", "net28") for r in rs if g(r, "test", "net28") is not None]),
                                    test_ex=med([r["test"]["ex"] for r in rs]), both_net28_pos=float(np.mean([(g(r, "train", "net28") or -1) > 0 and (g(r, "test", "net28") or -1) > 0 for r in rs]))) for v, rs in vals.items()]
    key = lambda r: min(g(r, "train", "net28") if g(r, "train", "net28") is not None else -9, g(r, "test", "net28") if g(r, "test", "net28") is not None else -9)
    out["top"] = sorted(full, key=key, reverse=True)[:8]
    return out


def family_grids(df):
    g = lambda k: np.nan_to_num(df[k].to_numpy())
    btc = df.btc_d_sma50.to_numpy()
    regimes = [("전체", np.ones(len(df), bool)), ("BTC 50일선 위", btc > 0), ("BTC 50일선 아래", btc <= 0)]
    rev = g("rev_bar") == 1
    wk = [("전체", np.ones(len(df), bool)), ("52주 하단 25%", df.w_pos52.to_numpy() < 0.25), ("주봉 상승 추세", (df.w_d_sma20.to_numpy() > 0) & (df.w_slope20.to_numpy() > 0))]
    # 가설 1: 거래량 터짐 + 신저가 → 저점 지킴
    rows = []
    for vm, ln, c, k, ctx in GRID1:
        base = g(f"g{vm}_{ln}_{c}_{k}_{ctx}") == 1
        for cname, cm in ((("없음", np.ones(len(df), bool)),) if k == 0 else (("없음", np.ones(len(df), bool)), ("전일 고가 넘는 양봉", rev))):
            for rname, rm in regimes:
                for wname, wm in wk:
                    m = dedup(df, base & cm & rm & wm)
                    rows.append(dict(p={"거래량 배수": f"{vm}배 이상", "신저가 기준": f"{ln}일", "당일 종가 위치": "범위 위쪽 60% 안" if c else "조건 없음", "저점 지킨 일수": f"{k}일" if k else "당일 바로", "하락 흐름 조건": "있음" if ctx else "없음", "확인 봉": cname, "BTC 장세": rname, "주봉 상태": wname}, **light_stats(df, m)))
    dims1 = [(d, None) for d in ("거래량 배수", "신저가 기준", "당일 종가 위치", "저점 지킨 일수", "하락 흐름 조건", "확인 봉", "BTC 장세", "주봉 상태")]
    out = dict(capitulation=summarize_family(rows, dims1))
    # 가설 2: 조용한 횡보 뒤 거래량 동반 돌파
    rows = []
    rng, quiet, vr = df.range60_prev.to_numpy(), df.days_since_pump.to_numpy(), df.vr.to_numpy()
    for rv in (0.3, 0.45, 0.6, 0.9, 99):
        for q in (0, 60, 120, 250):
            for b in ("brk20", "brk60", "brk250"):
                for v in (1.5, 2, 3, 5):
                    base = (rng < rv) & (quiet > q) & (g(b) == 1) & (vr >= v)
                    for rname, rm in regimes:
                        m = dedup(df, base & rm)
                        rows.append(dict(p={"직전 60일 범위": "제한 없음" if rv > 9 else f"{round(rv * 100)}% 미만", "급등 없던 기간": "제한 없음" if q == 0 else f"{q}일 넘게", "돌파 기준": {"brk20": "20일 고점", "brk60": "60일 고점", "brk250": "250일 고점"}[b], "거래량 배수": f"{v}배 이상", "BTC 장세": rname}, **light_stats(df, m)))
    dims2 = [(d, None) for d in ("직전 60일 범위", "급등 없던 기간", "돌파 기준", "거래량 배수", "BTC 장세")]
    out["base_breakout"] = summarize_family(rows, dims2)
    return out


def verdict(st):
    a, b = st["train"], st["test"]
    if a.get("n", 0) < 100 or b.get("n", 0) < 100:
        return "건수 부족"
    ok7 = (a.get("t") or 0) >= 2 and (b.get("t") or 0) >= 2 and b["net"] > 0
    ok28 = (a.get("t28") or 0) >= 2 and (b.get("t28") or 0) >= 2 and (b.get("net28") or -1) > 0
    if ok7 or ok28:
        return "통과"
    if a["net"] > 0 and b["net"] > 0 and a["ex"] > 0 and b["ex"] > 0:
        return "약함 (두 기간 플러스지만 통계적으로 뚜렷하지 않음)"
    if (a.get("t") or 0) <= -2 and (b.get("t") or 0) <= -2:
        return "반대 (또래보다 나쁨)"
    return "기준 미달"


# ----------------------------------------------------------------------------------------
# E. 시장 전체 신호 — "어느 종목"이 아니라 "언제"를 가리는 신호인지 확인
# ----------------------------------------------------------------------------------------
def market_signals(df):
    x = pd.DataFrame(dict(
        t=df.t.to_numpy(), fwd=df.fwd.to_numpy(), fwd28=df.fwd28.to_numpy(),
        brk=((df.brk60.to_numpy() > 0.5) & (df.vr.to_numpy() >= 2)).astype(float),
        base=((df.range60_prev.to_numpy() < 0.6) & (df.brk60.to_numpy() > 0.5) & (df.vr.to_numpy() >= 2)).astype(float),
        low=(df.newlow20.to_numpy() > 0.5).astype(float), test=df.is_test.to_numpy().astype(float), btc=df.btc_d_sma50.to_numpy(),
    ))
    day = x.groupby("t").agg(fwd=("fwd", "mean"), fwd28=("fwd28", "mean"), brk=("brk", "mean"), base=("base", "mean"), low=("low", "mean"), test=("test", "first"), btc=("btc", "first"), n=("fwd", "size")).reset_index()
    low5 = day.low.rolling(5).max().shift(1)
    sigs = [
        ("거래량 2배 동반 60일 고점 돌파 종목이 전체의 5% 이상", day.brk >= 0.05), ("같은 조건 10% 이상", day.brk >= 0.10), ("같은 조건 5% 이상 + BTC 50일선 위", (day.brk >= 0.05) & (day.btc > 0)),
        ("횡보(60일 범위 60% 미만) 뒤 돌파 종목이 전체의 2% 이상", day.base >= 0.02), ("같은 조건 2% 이상 + BTC 50일선 위", (day.base >= 0.02) & (day.btc > 0)),
        ("돌파 종목이 하나도 없는 날", day.brk == 0),
        ("20일 신저가 종목이 전체의 30% 이상 (동반 투매)", day.low >= 0.30), ("동반 투매(최근 5일 안에 30% 이상) 뒤 신저가 종목이 5% 미만으로 줄어든 날", (low5 >= 0.30) & (day.low < 0.05)),
        ("BTC 50일선 위", day.btc > 0), ("BTC 50일선 아래", day.btc <= 0),
    ]
    out = []
    for label, m in sigs:
        m = m.fillna(False).to_numpy()
        r = dict(label=label)
        for per, pm in (("train", day.test.to_numpy() == 0), ("test", day.test.to_numpy() == 1)):
            k = m & pm
            tt = day.t.to_numpy()[k]
            eps = int(1 + (np.diff(tt) > 10).sum()) if len(tt) else 0  # 10일 넘게 떨어져 있으면 다른 묶음으로 센다
            r[per] = dict(days=int(k.sum()), episodes=eps, share=float(k.sum() / max(pm.sum(), 1)), fwd=float(day.fwd[k].mean()) if k.sum() >= 5 else None, fwd28=float(day.fwd28[k].mean()) if k.sum() >= 5 else None,
                          base=float(day.fwd[pm].mean()), base28=float(day.fwd28[pm].mean()), t=hac_t(day.fwd[k].to_numpy() - day.fwd[pm].mean()) if k.sum() >= 40 else None)
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------
# B. 조건 조합 발굴
# ----------------------------------------------------------------------------------------
MKT_LABEL = {"btc_d_sma50": "BTC 50일선 대비 위치", "btc_roc7": "BTC 7일 등락률", "btc_roc28": "BTC 28일 등락률", "btc_rsi14": "BTC RSI", "breadth50": "50일선 위 코인 비율", "mkt_ret1": "시장 중앙값 1일 등락", "mkt_ret7": "시장 중앙값 7일 등락"}


def col_label(c):
    if c.startswith("r_"):
        return LABEL.get(c[2:], c[2:]) + " 순위"
    return LABEL3.get(c) or MKT_LABEL.get(c) or LABEL.get(c, c)


def is_flag(c):
    return c in BINARY or c in FLAGS3


def cond_text(c, thr, left):
    name = col_label(c)
    if thr < -100:
        return f"{name}: {'값 없음' if left else '값 있음'}"
    if c.startswith("r_"):
        return f"{name} 하위 {thr * 100:.0f}% 이내" if left else f"{name} 상위 {(1 - thr) * 100:.0f}% 이내"
    if is_flag(c) and 0 < thr < 1:
        return f"{name}: {'아니오' if left else '예'}"
    return f"{name} {'≤' if left else '>'} {thr:.3g}"


def mine(df, cols, target, depth=4, lag=7, rows=None):
    use = np.ones(len(df), bool) if rows is None else rows
    use = use & df[target].notna().to_numpy()
    tr = use & ~df.is_test.to_numpy()
    te = use & df.is_test.to_numpy()
    if tr.sum() < 3000 or te.sum() < 1000:
        return None
    X = np.nan_to_num(df[cols].to_numpy(dtype="float32"), nan=-999.0, posinf=-999.0, neginf=-999.0)
    y = df[target].clip(-0.3, 0.3).to_numpy()
    leaf_min = max(300, int(0.004 * tr.sum()))
    tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf_min, random_state=0).fit(X[tr], y[tr])
    T = tree.tree_
    leaf = np.full(len(df), -1)
    leaf[use] = tree.apply(X[use])
    paths = {}

    def walk(node, conds):
        if T.children_left[node] == -1:
            paths[node] = conds
            return
        c, thr = cols[T.feature[node]], float(T.threshold[node])
        walk(T.children_left[node], conds + [cond_text(c, thr, True)])
        walk(T.children_right[node], conds + [cond_text(c, thr, False)])

    walk(0, [])
    fwd = "fwd28" if target == "ex28" else "fwd"
    exc = target if target.startswith("ex") else "ex"
    out = []
    for node, conds in paths.items():
        r = dict(rule=conds)
        for per, pm in (("train", tr), ("test", te)):
            m = pm & (leaf == node)
            if m.sum() < 30:
                r[per] = dict(n=int(m.sum()))
                continue
            sub = df.loc[m, ["t", exc, fwd, "pump"]]
            d = sub.groupby("t")[exc].mean().to_numpy()
            r[per] = dict(n=int(m.sum()), days=int(sub.t.nunique()), ex=float(d.mean()), t=hac_t(d, lag) if len(d) >= 60 else None, net=float(sub[fwd].mean()), median=float(sub[fwd].median()), win=float((sub[fwd] > 0).mean()), pump=float(sub.pump.mean()), y=float(df.loc[m, target].clip(-0.3, 0.3).mean()))
        out.append(r)
    out.sort(key=lambda r: -(r["train"].get("y") or -9))
    return dict(target=target, leaves=len(out), best=out[:5], worst=out[-3:])


# ----------------------------------------------------------------------------------------
# C. 모델 재학습
# ----------------------------------------------------------------------------------------
def walk_forward(df, cols, target, embargo=S.H + 2, importance=True):
    X = df[cols].to_numpy(dtype="float32")
    X[~np.isfinite(X)] = np.nan
    y = df[target].to_numpy()
    years = df.date.str[:4].astype(int).to_numpy()
    t = df.t.to_numpy()
    pred = np.full(len(df), np.nan)
    last = None
    for yr in range(max(2021, years.min() + 1), years.max() + 1):
        te = years == yr
        if not te.any():
            continue
        tr = (years < yr) & (t <= t[te].min() - embargo) & np.isfinite(y)
        if tr.sum() < 30000:
            continue
        use = np.flatnonzero(np.isfinite(X[tr]).sum(axis=0) >= 500)
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=7)
        m.fit(X[tr][:, use], y[tr])
        pred[te] = m.predict_proba(X[te][:, use])[:, 1]
        last = (m, te, use)
        log(f"    {target} {yr}: 학습 {int(tr.sum()):,}행")
    imp = []
    if importance and last is not None:
        m, te, use = last
        idx = np.flatnonzero(te)
        idx = RNG.choice(idx, size=min(40000, len(idx)), replace=False)
        Xs, ys = X[idx][:, use], y[idx]

        def top_rate(Xm):
            p = m.predict_proba(Xm)[:, 1]
            return ys[p >= np.quantile(p, 0.95)].mean()

        base = top_rate(Xs)
        for f, ci in enumerate(use):
            Xp = Xs.copy()
            Xp[:, f] = RNG.permutation(Xp[:, f])
            imp.append((cols[ci], float(base - top_rate(Xp))))
        imp.sort(key=lambda x: -x[1])
    return pred, imp[:15]


def model_round(df, feats, cols3, baseline):
    base_cols = [f"r_{c}" for c in feats] + S.MARKET_FEATS
    cols = base_cols + cols3
    out = dict(rows=[])
    for target, name, bid in (("pump", "급등 확률 모델", "s_ml_pump@oos"), ("win5", "초과수익 모델", "s_ml_ret@oos")):
        pred, imp = walk_forward(df, cols, target)
        has = np.isfinite(pred)
        if has.sum() < 5000:
            continue
        need = pd.Series(has, index=df.index)
        picks = picks_of(df, pred, need)
        uni = df[has]
        r = dict(target=target, label=name, importance=[dict(label=col_label(k), new=k in cols3, drop=v) for k, v in imp], baseline=(baseline or {}).get(bid))
        for per, pm, um in (("train", ~picks.is_test, ~uni.is_test), ("test", picks.is_test, uni.is_test)):
            r[per] = trade_stats(picks[pm], uni[um], hac_t)
        r["topPicks"] = {k: int(v) for k, v in picks.sym[picks.is_test].value_counts().head(10).items()}
        out["rows"].append(r)
    return out


# ----------------------------------------------------------------------------------------
# D. 뉴스 × 차트
# ----------------------------------------------------------------------------------------
import html  # noqa: E402
import re  # noqa: E402

CATS = {
    "partner": ("제휴·도입", r"파트너십|제휴|협력|협업|MOU|맞손|손잡|통합|도입|채택|연동"),
    "upgrade": ("메인넷·업그레이드·출시", r"메인넷|업그레이드|하드포크|출시|론칭|런칭|테스트넷|로드맵"),
    "inst": ("ETF·기관 매수", r"ETF|기관|블랙록|그레이스케일|신탁|트레저리|매입|매수"),
    "burn": ("소각·바이백", r"소각|바이백|환매"),
    "fund": ("투자 유치", r"투자 ?유치|펀딩|시리즈 ?[A-C]|조달"),
    "listing": ("상장", r"상장|거래 ?지원|원화 ?마켓"),
    "delist": ("상장폐지·유의", r"상장 ?폐지|상폐|거래 ?지원 ?종료|유의 ?종목|투자 ?유의"),
    "hack": ("해킹·사고", r"해킹|탈취|익스플로잇|유출|공격|도난"),
    "legal": ("규제·소송", r"SEC|소송|기소|제재|규제|벌금|수사|패소|승소"),
    "whale": ("고래·대량 이체", r"고래|대량 ?이체|매집"),
    "unlock": ("에어드랍·언락", r"에어드랍|에어드롭|언락|락업 ?해제|물량 ?해제"),
    "price_up": ("가격 보도: 급등", r"급등|폭등|신고가|강세|랠리|치솟|상승"),
    "price_dn": ("가격 보도: 급락", r"급락|폭락|약세|붕괴|추락|신저가|하락"),
}
GOOD = ["partner", "upgrade", "inst", "burn", "fund"]
AMBIG = set("플로우 웨이브 하이브 에이스 스팀 메타 가스 썬 문 빔 원 비트코인 이더리움 테더 리퍼블릭 오리진 프로토콜 네트워크 토큰 코인 파이낸스 체인".split())


def load_news():
    d = os.path.join(NEWS, "blockmedia")
    if not os.path.isdir(d):
        return None
    rows = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                rows += json.load(f)
    if len(rows) < 1000:
        return None
    n = pd.DataFrame(rows, columns=["id", "ts", "title"]).drop_duplicates("id")
    n["title"] = n.title.map(html.unescape)
    n["ts"] = pd.to_datetime(n.ts, utc=True)
    return n


def tag_news(news, names):
    """제목에 코인 이름(한글 3자 이상) 또는 티커(영문 3자 이상, 앞뒤가 영문·숫자가 아님)가 있으면 그 코인의 뉴스로 본다."""
    kor = {}
    for sym, (ko, _) in names.items():
        if ko and len(ko) >= 3 and ko not in AMBIG and sym not in ("BTC", "ETH", "USDT"):
            kor[ko] = sym
    tick = [s for s in names if len(s) >= 3 and s.isalnum() and s not in ("BTC", "ETH", "USDT", "USD", "ETF", "SEC", "CEO", "NFT", "DAO", "GAS", "ONE", "ACE", "MOVE", "TOKEN", "FLOW", "HIVE", "META", "WAVES", "API", "KRW", "FED", "CPI", "AI")]
    re_k = re.compile("|".join(re.escape(k) for k in sorted(kor, key=len, reverse=True))) if kor else None
    re_t = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(s) for s in sorted(tick, key=len, reverse=True)) + r")(?![A-Za-z0-9])") if tick else None
    cat_re = {k: re.compile(v[1]) for k, v in CATS.items()}
    out = []
    for ts, title in zip(news.ts, news.title):
        syms = set()
        if re_k:
            syms.update(kor[m.group(0)] for m in re_k.finditer(title))
        if re_t:
            syms.update(m.group(1) for m in re_t.finditer(title))
        if not syms or len(syms) > 3:  # 여러 코인을 나열한 시황 기사는 뺀다
            continue
        cats = [k for k, r in cat_re.items() if r.search(title)]
        for s in syms:
            out.append((ts, s, "|".join(cats)))
    return pd.DataFrame(out, columns=["ts", "sym", "cats"])


def news_round(df, coins, cfg, tagged, cols_tree):
    tz = cfg["tz"]
    tg = tagged.copy()
    tg["date"] = (tg.ts + pd.Timedelta(hours=tz)).dt.strftime("%Y-%m-%d")
    ev = {}
    for k in CATS:
        e = tg[tg.cats.str.contains(k, regex=False)].groupby(["sym", "date"]).size().rename(f"n_{k}")
        ev[k] = e
    allc = tg.groupby(["sym", "date"]).size().rename("n_news")
    E = pd.concat([allc] + list(ev.values()), axis=1).fillna(0).reset_index()
    d = df.merge(E, on=["sym", "date"], how="left")
    for c in E.columns[2:]:
        d[c] = d[c].fillna(0)
    d["good"] = (d[[f"n_{k}" for k in GOOD]].sum(axis=1) > 0) & (d.n_delist == 0) & (d.n_hack == 0)
    # 그 코인에 지난 14일간 뉴스가 없었는가 (새 소식)
    has = (d.n_news > 0).astype(float)  # 표는 이미 코인별·날짜순으로 정렬돼 있다
    prev = has.groupby(d.sym).transform(lambda s: s.shift(1).rolling(14, min_periods=1).sum())
    d["fresh"] = (d.n_news > 0) & ~(prev > 0)
    out = dict(titles=int(len(tagged)), eventDays=int((d.n_news > 0).sum()), coins=int(d.sym[d.n_news > 0].nunique()), cats=[], states=[], sameDay=[])
    es = lambda m: event_stats(d, np.asarray(m, bool), coins, cfg["cost"], with_exits=False)
    out["cats"].append(dict(id="any", label="뉴스가 있던 날 (종류 무관)", **es(d.n_news > 0)))
    out["cats"].append(dict(id="fresh", label="14일 만에 처음 나온 뉴스", **es(d.fresh)))
    out["cats"].append(dict(id="good", label="호재성 뉴스 (제휴·출시·기관·소각·투자유치)", **es(d.good)))
    out["cats"].append(dict(id="good_fresh", label="호재성 뉴스 + 14일 만에 처음", **es(d.good & d.fresh)))
    for k, (lab, _) in CATS.items():
        m = (d[f"n_{k}"] > 0).to_numpy()
        st = es(m)
        st.update(id=k, label=lab, roc1=float(d.roc1[m].median()) if m.sum() >= 30 else None)
        out["cats"].append(st)
    # 호재성 뉴스 당일의 차트 상태별
    gm = d.good.to_numpy()
    g = lambda k: d[k].to_numpy()
    wup = (g("w_d_sma20") > 0) & (g("w_slope20") > 0)
    states = [
        ("이미 오른 뒤 (3일간 +15% 이상)", g("roc3") >= 0.15), ("아직 안 움직임 (3일 등락 ±5% 이내)", np.abs(g("roc3")) <= 0.05), ("빠진 상태 (3일간 −10% 이하)", g("roc3") <= -0.10),
        ("뉴스 당일 +10% 이상 급등", g("roc1") >= 0.10), ("뉴스 당일 등락 ±3% 이내", np.abs(g("roc1")) <= 0.03),
        ("주봉 상승 추세", wup), ("주봉 하락 추세", (g("w_d_sma20") < 0) & (g("w_slope20") < 0)),
        ("52주 범위 하단 25%", g("w_pos52") < 0.25), ("52주 범위 상단 25%", g("w_pos52") > 0.75),
        ("당일 거래량 3배 이상", g("vr") >= 3), ("당일 거래량 평소 수준 (1.5배 미만)", g("vr") < 1.5),
        ("투매 뒤 저점을 지키는 중", (g("dsc") <= 10) & (g("hold") >= 0)), ("일봉 저점 높이는 중", g("hl") == 1),
        ("BTC 50일선 위", g("btc_d_sma50") > 0), ("BTC 50일선 아래", g("btc_d_sma50") <= 0),
        ("주봉 상승 추세 + 아직 안 움직임 + BTC 50일선 위", wup & (np.abs(g("roc3")) <= 0.05) & (g("btc_d_sma50") > 0)),
        ("52주 하단 25% + 당일 거래량 3배 이상", (g("w_pos52") < 0.25) & (g("vr") >= 3)),
    ]
    for lab, m in states:
        out["states"].append(dict(label=lab, **es(gm & np.nan_to_num(m.astype(float)).astype(bool))))
    # 호재성 뉴스가 나온 날들만 놓고 조건 조합 발굴
    out["tree"] = mine(d, cols_tree, "ex", depth=3, rows=gm)
    return out


# ----------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    raw = {m: S.load_market(m, cfg) for m, cfg in S.MARKETS.items() if os.path.isdir(f"{S.DATA}/{m}")}
    names = {}
    if os.path.isdir(f"{S.DATA}/bithumb"):
        for fn in os.listdir(f"{S.DATA}/bithumb"):
            with open(f"{S.DATA}/bithumb/{fn}", encoding="utf-8") as f:
                j = json.load(f)
            names[j["sym"]] = (j.get("name"), j.get("en"))
    news = load_news()
    tagged = None
    if news is not None:
        for m in raw:
            for s in raw[m]:
                names.setdefault(s, (None, None))
        tagged = tag_news(news, names)
        log(f"뉴스 제목 {len(news):,}건 ({news.ts.min():%Y-%m} ~ {news.ts.max():%Y-%m}) 중 코인이 잡힌 것 {len(tagged):,}건")
    baseline = {}
    try:
        with open(f"{OUT}/indicators.json", encoding="utf-8") as f:
            for m in json.load(f)["markets"]:
                baseline[m["market"]] = {s["id"]: s["periods"] for s in m["scores"]}
    except Exception:
        pass
    result = dict(generatedAt=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M"), markets=[], news=None if news is None else dict(titles=int(len(news)), first=str(news.ts.min())[:10], last=str(news.ts.max())[:10], tagged=int(len(tagged))))
    for mkt, cfg in S.MARKETS.items():
        if mkt not in raw or "BTC" not in raw[mkt]:
            continue
        raw.setdefault("binance" if mkt == "bithumb" else "bithumb", {})
        log(f"[{cfg['label']}] 지표 계산")
        df, feats, all_dates, excluded = S.build(mkt, cfg, raw)
        df, cols3 = add_features3(df, raw[mkt])
        model3 = [c for c in cols3 if c in MODEL3]
        log(f"  표 {len(df):,}행 · 일봉 지표 {len(feats)}개 + 주봉·순서형 {len(model3)}개")
        D = setups(df)
        rows = []
        for sid, (group, label, mask) in D.items():
            m = dedup(df, np.nan_to_num(np.asarray(mask, float)).astype(bool))
            st = event_stats(df, m, raw[mkt], cfg["cost"])
            rows.append(dict(id=sid, group=group, label=label, verdict=verdict(st), **st))
        log(f"  A 시나리오 {len(rows)}개 채점 완료")
        mkt_sig = market_signals(df)
        tree_cols = [c if c in BINARY else f"r_{c}" for c in feats] + model3
        fam = family_grids(df)
        log("  A2 가설 묶음 (기준값을 바꿔 가며) 채점 완료")
        mines = dict(
            ex7=mine(df, tree_cols, "ex"), ex28=mine(df, tree_cols, "ex28", lag=28), abs7=mine(df, tree_cols + S.MARKET_FEATS, "fwd"),
            new_only=mine(df, model3, "ex"),
        )
        log("  B 조건 조합 발굴 완료")
        models = model_round(df, feats, model3, baseline.get(mkt))
        log("  C 모델 재학습 완료")
        nr = None
        if tagged is not None:
            nr = news_round(df, raw[mkt], cfg, tagged, tree_cols)
            log(f"  D 뉴스 × 차트 완료 (뉴스가 붙은 코인·날 {nr['eventDays']:,}개)")
        result["markets"].append(dict(market=mkt, label=cfg["label"], rows=int(len(df)), coins=int(df.sym.nunique()), dates=[df.date.min(), df.date.max()], base=dict(train=float(df.pump[~df.is_test].mean()), test=float(df.pump[df.is_test].mean()), net_train=float(df.fwd[~df.is_test].mean()), net_test=float(df.fwd[df.is_test].mean()), net28_train=float(df.fwd28[~df.is_test].mean()), net28_test=float(df.fwd28[df.is_test].mean()), win_train=float((df.fwd[~df.is_test] > 0).mean()), win_test=float((df.fwd[df.is_test] > 0).mean())), setups=rows, families=fam, marketSignals=mkt_sig, mines=mines, models=models, news=nr))
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
    with open(f"{OUT}/setups.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    from report3 import render

    with open(f"{OUT}/SETUPS.md", "w", encoding="utf-8") as f:
        f.write(render(result))
    log(f"끝 ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
