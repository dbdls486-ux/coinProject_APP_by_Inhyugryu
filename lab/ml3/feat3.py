"""
3차 지표 — 주봉 지표와 '순서가 있는' 패턴 지표.

  주봉: 날마다 "오늘로 끝나는 7일"을 한 주로 묶는다 (달력 주가 아니라 굴러가는 주).
        그러면 어느 날에 봐도 그날까지의 정보만 쓰고, 주가 끝나길 기다릴 필요도 없다.
  순서형: 투매(거래량 폭발 + 신저가) 뒤 저점이 지켜지는지, 저점 재확인, 고점·저점 구조 전환,
          매집 흔적, BTC 조정 때 버틴 정도, 고점 돌파 뒤 첫 눌림 등.

모든 값은 그날 종가까지의 정보로만 계산한다. 고점·저점(피벗)은 앞뒤 3일을 봐야 확정되므로 3일 늦게 반영한다.
"""
import numpy as np
import pandas as pd

swv = np.lib.stride_tricks.sliding_window_view
WK = 7
PIV = 3  # 피벗 확정에 필요한 양옆 일수


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def weekly(o, h, l, c, v):
    n = len(c)
    H, L, V = pd.Series(h), pd.Series(l), pd.Series(v)
    wh = H.rolling(WK).max().to_numpy()
    wl = L.rolling(WK).min().to_numpy()
    wv = V.rolling(WK).sum().to_numpy()
    cols = ["w_d_sma10", "w_d_sma20", "w_d_sma40", "w_slope20", "w_align", "w_gc", "w_rsi", "w_rsi_turn", "w_macd_hist", "w_macd_up", "w_macd_cross", "w_bb_pctb", "w_bb_width_pct",
            "w_pos52", "w_d_high52", "w_d_low52", "w_vol_trend", "w_up8", "w_hl2", "w_ll3", "w_ret12", "w_ret26"]
    out = {k: np.full(n, np.nan, dtype="float32") for k in cols}
    for p in range(WK):
        idx = np.arange(p, n, WK)
        if len(idx) < 12:
            continue
        ok = idx >= WK - 1
        wc = pd.Series(np.where(ok, c[idx], np.nan))
        xh, xl, xv = pd.Series(wh[idx]), pd.Series(wl[idx]), pd.Series(wv[idx])
        s10, s20, s40 = wc.rolling(10).mean(), wc.rolling(20).mean(), wc.rolling(40).mean()
        r = {}
        r["w_d_sma10"], r["w_d_sma20"], r["w_d_sma40"] = wc / s10 - 1, wc / s20 - 1, wc / s40 - 1
        r["w_slope20"] = s20 / s20.shift(4) - 1
        r["w_align"] = ((wc > s10) & (s10 > s20) & (s20 > s40)).astype(float).where(s40.notna())
        r["w_gc"] = ((s10 > s40) & (s10.shift(8) <= s40.shift(8))).astype(float).where(s40.shift(8).notna())
        rsi = _rsi(wc, 14)
        r["w_rsi"] = rsi
        r["w_rsi_turn"] = ((rsi.rolling(4).min() < 35) & (rsi > rsi.shift(1))).astype(float).where(rsi.notna())
        macd = wc.ewm(span=12, adjust=False, min_periods=12).mean() - wc.ewm(span=26, adjust=False, min_periods=26).mean()
        hist = macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()
        r["w_macd_hist"] = hist / wc
        r["w_macd_up"] = (hist > hist.shift(1)).astype(float).where(hist.shift(1).notna())
        r["w_macd_cross"] = ((hist > 0) & (hist.shift(1) <= 0)).astype(float).where(hist.shift(1).notna())
        sd = wc.rolling(20).std()
        r["w_bb_pctb"] = (wc - (s20 - 2 * sd)) / (4 * sd).replace(0, np.nan)
        r["w_bb_width_pct"] = (4 * sd / s20).rolling(52, min_periods=26).rank(pct=True)
        hi, lo = xh.rolling(52, min_periods=26).max(), xl.rolling(52, min_periods=26).min()
        r["w_pos52"] = (wc - lo) / (hi - lo).replace(0, np.nan)
        r["w_d_high52"], r["w_d_low52"] = wc / hi - 1, wc / lo - 1
        r["w_vol_trend"] = xv.rolling(4).mean() / xv.rolling(20).mean().replace(0, np.nan)
        r["w_up8"] = (wc > wc.shift(1)).astype(float).rolling(8).sum()
        r["w_hl2"] = ((xl > xl.shift(1)) & (xl.shift(1) > xl.shift(2))).astype(float).where(xl.shift(2).notna())
        r["w_ll3"] = ((xl < xl.shift(1)) & (xl.shift(1) < xl.shift(2)) & (xl.shift(2) < xl.shift(3))).astype(float).where(xl.shift(3).notna())
        r["w_ret12"], r["w_ret26"] = wc / wc.shift(12) - 1, wc / wc.shift(26) - 1
        for k in cols:
            out[k][idx] = r[k].to_numpy()
    # 주봉 피보나치: 최근 26주(182일) 안의 최저점 → 그 뒤 최고점까지를 상승 파동으로 보고 지금 되돌림 비율
    W = 26 * WK
    retr = np.full(n, np.nan, dtype="float32")
    size = np.full(n, np.nan, dtype="float32")
    if n >= W:
        Wl, Wh = swv(l, W), swv(h, W)
        il = Wl.argmin(axis=1)
        lowp = Wl[np.arange(len(il)), il]
        after = np.arange(W)[None, :] >= il[:, None]
        hip = np.where(after, Wh, -np.inf).max(axis=1)
        swing = hip / lowp - 1
        cc = c[W - 1 :]
        rr = np.where(hip > lowp, (hip - cc) / (hip - lowp), np.nan)
        retr[W - 1 :] = np.where(swing >= 0.5, rr, np.nan)  # +50% 이상 오른 파동만
        size[W - 1 :] = swing
    out["w_fib_retr"] = retr
    out["w_fib_zone"] = np.where(np.isfinite(retr), ((retr >= 0.342) & (retr <= 0.658)).astype("float32"), 0).astype("float32")
    out["w_swing"] = size
    return out


def sequence(o, h, l, c, v, btc_c):
    n = len(c)
    O, H, L, C, V = (pd.Series(x) for x in (o, h, l, c, v))
    out = {}
    v20 = V.rolling(20).mean().shift(1)
    vr = (V / v20.replace(0, np.nan)).to_numpy()
    rng = (H - L).replace(0, np.nan)
    clv = ((C - L) / rng).fillna(0.5).to_numpy()
    ll20 = L.rolling(20).min().shift(1)
    newlow = (L < ll20).to_numpy()
    newlows10 = pd.Series(newlow.astype(float)).rolling(10).sum().to_numpy()
    drop10 = (C / C.shift(10) - 1).to_numpy()
    decline = (newlows10 >= 3) | (drop10 <= -0.15)
    # 투매 뒤 버팀의 출발점: 하락 흐름 속에서 거래량이 평소의 3배 이상 터졌고 20일 신저가를 찍었지만 종가가 그날 범위 아래쪽 40% 위에서 끝남
    clx = (vr >= 3) & newlow & (clv >= 0.4) & decline
    clx_any = (vr >= 3) & newlow & decline  # 종가 위치 조건 없이
    out["vr"], out["clv"], out["newlows10"], out["drop10"] = vr, clv, newlows10, drop10
    out["clx"], out["clx_any"] = clx.astype(float), clx_any.astype(float)

    # 투매일 이후 상태 (최대 30일 추적)
    dsc = np.full(n, np.nan)
    hold = np.full(n, np.nan)
    rsc = np.full(n, np.nan)
    vol_after = np.full(n, np.nan)
    retest = np.zeros(n)
    clx_low_arr = np.full(n, np.nan)
    last, lowp, minlow, vsum, cv = -1, np.nan, np.inf, 0.0, np.nan
    for i in range(n):
        if clx[i]:
            last, lowp, minlow, vsum, cv = i, l[i], np.inf, 0.0, v[i]
            dsc[i], hold[i], rsc[i] = 0, 0.0, 0.0
            clx_low_arr[i] = lowp
            continue
        if last >= 0 and i - last <= 30:
            k = i - last
            minlow = min(minlow, l[i])
            vsum += vr[i] if np.isfinite(vr[i]) else 0.0
            dsc[i] = k
            hold[i] = minlow / lowp - 1
            rsc[i] = c[i] / c[last] - 1
            vol_after[i] = vsum / k
            clx_low_arr[i] = lowp
            # 저점 재확인: 투매 5일 이후, 최근 3일 저가가 투매 저점의 −3%~+5% 안, 그 3일 거래량이 투매일의 절반 미만
            if k >= 5:
                lo3 = l[max(i - 2, last + 1) : i + 1].min()
                v3 = v[max(i - 2, last + 1) : i + 1].max()
                if lowp * 0.97 <= lo3 <= lowp * 1.05 and v3 < 0.5 * cv and minlow >= lowp * 0.97:
                    retest[i] = 1
    out["dsc"], out["hold"], out["rsc"], out["vol_after"], out["retest"], out["clx_low"] = dsc, hold, rsc, vol_after, retest, clx_low_arr

    # 고점·저점 구조 (피벗은 3일 뒤 확정)
    hl = np.full(n, np.nan)
    hh = np.full(n, np.nan)
    turn = np.zeros(n)
    d_piv_low = np.full(n, np.nan)
    brk_piv = np.zeros(n)
    if n > 2 * PIV + 1:
        wl_, wh_ = swv(l, 2 * PIV + 1), swv(h, 2 * PIV + 1)
        is_pl = np.zeros(n, bool)
        is_ph = np.zeros(n, bool)
        is_pl[PIV : n - PIV] = l[PIV : n - PIV] == wl_.min(axis=1)
        is_ph[PIV : n - PIV] = h[PIV : n - PIV] == wh_.max(axis=1)
        pls, phs = [], []
        for i in range(n):
            j = i - PIV  # 오늘 확정되는 피벗
            new_low = False
            if j >= 0:
                if is_pl[j]:
                    pls.append(l[j])
                    new_low = True
                if is_ph[j]:
                    phs.append(h[j])
            if len(pls) >= 2:
                hl[i] = float(pls[-1] > pls[-2])
                d_piv_low[i] = c[i] / pls[-1] - 1
                if new_low and len(pls) >= 3 and pls[-1] > pls[-2] and pls[-2] < pls[-3]:
                    turn[i] = 1  # 저점을 낮추다가 처음으로 저점을 높인 것이 확정된 날
            if len(phs) >= 2:
                hh[i] = float(phs[-1] > phs[-2])
            if phs and i > 0 and c[i] > phs[-1] and c[i - 1] <= phs[-1]:
                brk_piv[i] = 1  # 직전 확정 고점을 종가로 넘음
    out["hl"], out["hh"], out["struct_turn"], out["d_piv_low"], out["brk_piv"] = hl, hh, turn, d_piv_low, brk_piv
    out["turn10"] = pd.Series(turn).rolling(10).max().to_numpy()

    # 방아쇠 봉
    rev = ((C > H.shift(1)) & (C > O)).to_numpy()
    out["rev_bar"] = rev.astype(float)
    out["brk10"] = (C > H.rolling(10).max().shift(1)).astype(float).to_numpy()
    rsi = _rsi(C, 14)
    out["rsi_min5"] = rsi.rolling(5).min().to_numpy()
    out["rsi_up40"] = ((rsi > 40) & (rsi.shift(1) <= 40)).astype(float).to_numpy()
    out["vr_max5"] = pd.Series(vr).rolling(5).max().to_numpy()
    sma20, sma50 = C.rolling(20).mean(), C.rolling(50).mean()
    out["near_sma20"] = ((C / sma20 - 1).abs() < 0.03).astype(float).to_numpy()
    out["touch50_5"] = ((L / sma50 - 1).abs() < 0.03).astype(float).rolling(5).max().to_numpy()
    sd20 = C.rolling(20).std()
    width = 4 * sd20 / sma20
    out["sq5"] = (width.rolling(120, min_periods=60).rank(pct=True).rolling(5).min().shift(1) < 0.15).astype(float).to_numpy()

    # 횡보·매집
    out["range60_prev"] = (H.rolling(60).max() / L.rolling(60).min() - 1).shift(1).to_numpy()
    out["vol_trend"] = (V.rolling(20).mean() / V.rolling(120).mean().replace(0, np.nan)).to_numpy()
    up = (C > C.shift(1)).astype(float)
    out["upvol10"] = ((V * up).rolling(10).sum() / V.rolling(10).sum().replace(0, np.nan)).to_numpy()
    out["absorb10"] = pd.Series(((vr >= 2) & (np.abs(c / o - 1) < 0.03)).astype(float)).rolling(10).sum().to_numpy()
    out["pos60_prev"] = ((C - L.rolling(60).min()) / (H.rolling(60).max() - L.rolling(60).min()).replace(0, np.nan)).shift(1).to_numpy()

    # 250일 고점 돌파 이후 경과일
    brk250 = (C > H.rolling(250).max().shift(1)).to_numpy()
    since = np.full(n, np.nan)
    last = -1
    for i in range(n):
        if brk250[i]:
            last = i
        if last >= 0:
            since[i] = i - last
    out["since_brk250"] = since

    # BTC 조정 때 버틴 정도
    B = pd.Series(btc_c)
    dd30 = C / H.rolling(30).max() - 1
    bdd30 = B / B.rolling(30).max() - 1
    out["dd_gap"] = (dd30 - bdd30).to_numpy()
    out["btc_dd_recent"] = ((B / B.shift(14) - 1).rolling(10).min() <= -0.10).astype(float).to_numpy()
    b20 = B.rolling(20).mean()
    below = (B < b20).astype(float)
    cross = ((B > b20) & (B.shift(1) <= b20.shift(1)) & (below.shift(1).rolling(10).sum() >= 7)).astype(float)
    out["btc_reclaim"] = cross.rolling(3).max().to_numpy()  # 10일 중 7일 이상 20일선 아래 있다가 회복한 지 3일 이내

    # 구조 기반 손절 기준: 최근 10일 저가
    out["stop_ref"] = L.rolling(10).min().to_numpy()
    return out


LABEL3 = {
    "w_d_sma10": "주봉 10주선 대비 위치", "w_d_sma20": "주봉 20주선 대비 위치", "w_d_sma40": "주봉 40주선 대비 위치", "w_slope20": "주봉 20주선 기울기", "w_align": "주봉 정배열",
    "w_gc": "주봉 골든크로스 10·40주 (최근 8주)", "w_rsi": "주봉 RSI", "w_rsi_turn": "주봉 RSI 과매도 뒤 반등", "w_macd_hist": "주봉 MACD 히스토그램", "w_macd_up": "주봉 MACD 히스토그램 상승",
    "w_macd_cross": "주봉 MACD 골든크로스", "w_bb_pctb": "주봉 볼린저 %B", "w_bb_width_pct": "주봉 볼린저 밴드폭 백분위", "w_pos52": "52주 범위 내 위치", "w_d_high52": "52주 고점 대비",
    "w_d_low52": "52주 저점 대비", "w_vol_trend": "주봉 거래량 추세 (4주/20주)", "w_up8": "최근 8주 중 상승한 주 수", "w_hl2": "주봉 저점 2주 연속 높임", "w_ll3": "주봉 저점 3주 연속 낮춤",
    "w_ret12": "12주 등락률", "w_ret26": "26주 등락률", "w_fib_retr": "주봉 피보나치 되돌림 비율", "w_fib_zone": "주봉 피보나치 0.382~0.618 구간", "w_swing": "26주 상승 파동 크기",
    "vr": "거래량 배수 (오늘/20일)", "clv": "종가의 당일 범위 내 위치", "newlows10": "최근 10일 중 신저가 일수", "drop10": "10일 등락률", "clx": "투매 뒤 버팀 (거래량 3배 + 신저가 + 종가 회복)",
    "clx_any": "거래량 3배 + 신저가", "dsc": "투매일 이후 경과일", "hold": "투매 저점 대비 이후 최저가", "rsc": "투매일 종가 대비 수익", "vol_after": "투매 이후 평균 거래량 배수",
    "retest": "투매 저점 재확인 (거래량 줄며)", "hl": "저점 높임", "hh": "고점 높임", "struct_turn": "저점 낮추다 처음 높임", "turn10": "최근 10일 안에 저점 구조 전환", "d_piv_low": "직전 저점 대비 위치",
    "brk_piv": "직전 고점 돌파", "rev_bar": "전일 고가 넘는 양봉", "brk10": "10일 고점 돌파", "rsi_min5": "최근 5일 RSI 최저", "rsi_up40": "RSI 40 상향 돌파", "vr_max5": "최근 5일 최대 거래량 배수",
    "near_sma20": "20일선 ±3% 이내", "touch50_5": "최근 5일 안에 50일선 근처까지 눌림", "sq5": "직전 5일 안에 밴드폭 수축", "range60_prev": "직전 60일 가격 범위", "vol_trend": "거래량 추세 (20일/120일)",
    "upvol10": "최근 10일 상승일 거래량 비중", "absorb10": "거래량 2배인데 가격 제자리인 날 수 (10일)", "pos60_prev": "직전 60일 범위 내 위치", "since_brk250": "250일 고점 돌파 이후 경과일",
    "dd_gap": "30일 고점 대비 낙폭이 BTC보다 작은 정도", "btc_dd_recent": "최근 BTC 14일 −10% 조정 있었음", "btc_reclaim": "BTC가 20일선 회복",
}
# 모델·나무에 넣을 지표 (손절 기준가 등 가격 수준 자체는 뺀다)
MODEL3 = [k for k in LABEL3 if k not in ("clx_low",)]


def features3(o, h, l, c, v, btc_c):
    d = weekly(o, h, l, c, v)
    d.update(sequence(o, h, l, c, v, btc_c))
    return pd.DataFrame({k: np.asarray(x, dtype="float32") for k, x in d.items()})
