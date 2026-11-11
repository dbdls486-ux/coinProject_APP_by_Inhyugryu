"""
지표 계산 — 한 코인의 일봉으로 지표 100여 개를 만든다.
모든 값은 그날 종가까지의 정보로만 계산한다 (미래 정보 없음).
목표값(앞으로 7일의 수익·급등 여부)은 targets() 에서 따로 만든다.
"""
import numpy as np
import pandas as pd

swv = np.lib.stride_tricks.sliding_window_view


def _wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _rsi(c, n):
    d = c.diff()
    up = _wilder(d.clip(lower=0), n)
    dn = _wilder((-d).clip(lower=0), n)
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.where(dn != 0, 100.0).where(up.notna())


def _days_since(x, n, mode):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = swv(x, n)
        idx = np.argmax(w, axis=1) if mode == "max" else np.argmin(w, axis=1)
        out[n - 1 :] = n - 1 - idx
    return out


def _recent(flag, n=3):
    """신호가 최근 n일 안에 나왔는가"""
    return flag.astype(float).rolling(n, min_periods=1).max()


def _streak(cond):
    grp = (~cond).cumsum()
    return cond.astype(int).groupby(grp).cumsum().astype(float)


def features(o, h, l, c, v, btc_c, extra=None):
    """o,h,l,c,v: numpy 배열. btc_c: 같은 날짜에 맞춘 BTC 종가. extra: {'prem':..., 'fund':...} (없으면 NaN)"""
    n = len(c)
    O, H, L, C, V = (pd.Series(x, dtype="float64") for x in (o, h, l, c, v))
    F = {}
    pc = C.shift(1)
    ret1 = C / pc - 1
    logret = np.log(C / pc)

    # ---------- 이동평균·추세 ----------
    sma = {k: C.rolling(k).mean() for k in (5, 10, 20, 50, 60, 100, 200)}
    for k in (5, 10, 20, 50, 100, 200):
        F[f"d_sma{k}"] = C / sma[k] - 1
    F["slope_sma20"] = sma[20] / sma[20].shift(5) - 1
    F["slope_sma50"] = sma[50] / sma[50].shift(10) - 1
    F["ma_align"] = ((C > sma[20]) & (sma[20] > sma[50]) & (sma[50] > sma[100])).astype(float)
    F["ma_align_dn"] = ((C < sma[20]) & (sma[20] < sma[50]) & (sma[50] < sma[100])).astype(float)
    for a, b in ((5, 20), (20, 60), (50, 200)):
        up = (sma[a] > sma[b]) & (sma[a].shift(1) <= sma[b].shift(1))
        dn = (sma[a] < sma[b]) & (sma[a].shift(1) >= sma[b].shift(1))
        F[f"gc_{a}_{b}"] = _recent(up)
        F[f"dc_{a}_{b}"] = _recent(dn)
    e12 = C.ewm(span=12, adjust=False, min_periods=12).mean()
    e26 = C.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = (e12 - e26) / C
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd - sig
    F["macd"] = macd
    F["macd_hist"] = hist
    F["macd_cross_up"] = _recent((hist > 0) & (hist.shift(1) <= 0))
    F["macd_cross_up_neg"] = _recent((hist > 0) & (hist.shift(1) <= 0) & (macd < 0))  # 0선 아래에서의 골든크로스

    tr = pd.concat([H - L, (H - pc).abs(), (L - pc).abs()], axis=1).max(axis=1)
    atr14 = _wilder(tr, 14)
    upm = H.diff()
    dnm = -L.diff()
    pdm = upm.where((upm > dnm) & (upm > 0), 0.0)
    ndm = dnm.where((dnm > upm) & (dnm > 0), 0.0)
    pdi = 100 * _wilder(pdm, 14) / atr14
    ndi = 100 * _wilder(ndm, 14) / atr14
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    F["adx14"] = _wilder(dx, 14)
    F["di_diff"] = pdi - ndi

    # 일목균형표: 구름은 26일 전에 계산된 값을 오늘 자리에 놓는다
    hh = {k: H.rolling(k).max() for k in (9, 10, 14, 20, 26, 52, 60, 120, 250)}
    ll = {k: L.rolling(k).min() for k in (9, 10, 14, 20, 26, 52, 60, 120, 250)}
    tenkan = (hh[9] + ll[9]) / 2
    kijun = (hh[26] + ll[26]) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hh[52] + ll[52]) / 2).shift(26)
    top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    F["ichi_cloud"] = (C > top).astype(float) - (C < bot).astype(float)
    F["ichi_d_cloud"] = C / top - 1
    F["ichi_tk"] = tenkan / kijun - 1
    F["ichi_tk_cross"] = _recent((tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1)))
    F["ichi_break_up"] = _recent((C > top) & (pc <= top.shift(1)))  # 구름 상향 돌파

    # ---------- 모멘텀·오실레이터 ----------
    rsi14 = _rsi(C, 14)
    F["rsi14"] = rsi14
    F["rsi2"] = _rsi(C, 2)
    F["rsi_cross30"] = _recent((rsi14 > 30) & (rsi14.shift(1) <= 30))
    low20c = C.rolling(20).min()
    F["bull_div"] = ((C <= low20c * 1.01) & (rsi14 - rsi14.rolling(20).min() >= 5)).astype(float)  # 가격은 저점인데 RSI는 저점을 높임
    k_fast = 100 * (C - ll[14]) / (hh[14] - ll[14]).replace(0, np.nan)
    k_slow = k_fast.rolling(3).mean()
    d_slow = k_slow.rolling(3).mean()
    F["stoch_k"] = k_slow
    F["stoch_kd"] = k_slow - d_slow
    F["stoch_cross_os"] = _recent((k_slow > d_slow) & (k_slow.shift(1) <= d_slow.shift(1)) & (k_slow < 25))
    F["willr14"] = -100 * (hh[14] - C) / (hh[14] - ll[14]).replace(0, np.nan)
    tp = (H + L + C) / 3
    cci = np.full(n, np.nan)
    if n >= 20:
        w = swv(tp.to_numpy(), 20)
        m = w.mean(axis=1)
        mad = np.abs(w - m[:, None]).mean(axis=1)
        cci[19:] = (w[:, -1] - m) / (0.015 * np.where(mad == 0, np.nan, mad))
    F["cci20"] = pd.Series(cci)
    mf = tp * V
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    F["mfi14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    roc = {k: C / C.shift(k) - 1 for k in (1, 3, 7, 14, 28, 60, 120)}
    for k, s in roc.items():
        F[f"roc{k}"] = s

    # ---------- 변동성·수렴 ----------
    F["atr_pct"] = atr14 / C
    F["atr_ratio"] = tr.rolling(5).mean() / tr.rolling(60).mean()
    sd20 = C.rolling(20).std(ddof=0)
    F["bb_pctb"] = (C - (sma[20] - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    bbw = 4 * sd20 / sma[20]
    F["bb_width"] = bbw
    F["bb_width_pct"] = bbw.rolling(120, min_periods=60).rank(pct=True)  # 낮을수록 밴드가 좁아진 상태(스퀴즈)
    atr20 = tr.rolling(20).mean()
    F["kc_squeeze"] = (2 * sd20 < 1.5 * atr20).astype(float).where(sd20.notna())
    rng = H - L
    F["nr7"] = (rng <= rng.rolling(7).min()).astype(float)
    F["range10"] = (hh[10] - ll[10]) / C
    rv20 = logret.rolling(20).std()
    F["rv_pct"] = rv20.rolling(250, min_periods=120).rank(pct=True)

    # ---------- 거래량 ----------
    vprev = V.rolling(20).mean().shift(1)
    F["vol_ratio"] = V / vprev.replace(0, np.nan)
    v5_60 = V.rolling(5).mean() / V.rolling(60).mean().replace(0, np.nan)
    F["vol5_60"] = v5_60
    F["vol10_30"] = V.rolling(10).mean() / V.rolling(30).mean().replace(0, np.nan)
    obv = (np.sign(C.diff()).fillna(0) * V).cumsum()
    F["obv_slope20"] = (obv - obv.shift(20)) / V.rolling(20).sum().replace(0, np.nan)
    upv = V.where(C > pc, 0.0).rolling(20).sum()
    dnv = V.where(C < pc, 0.0).rolling(20).sum()
    F["updown_vol20"] = np.log((upv + 1) / (dnv + 1))
    val20 = V.rolling(20).mean()
    F["val20_log"] = np.log10(val20.replace(0, np.nan))
    F["accum"] = ((v5_60 > 1.5) & (roc[7].abs() < 0.05)).astype(float)  # 가격은 조용한데 거래량만 늘어남
    F["vol_dry"] = (v5_60 < 0.5).astype(float)

    # ---------- 가격 구조 ----------
    for k in (20, 60, 120, 250):
        F[f"d_high{k}"] = C / hh[k] - 1
    for k in (20, 60, 250):
        F[f"d_low{k}"] = C / ll[k] - 1
    F["brk20"] = (C > hh[20].shift(1)).astype(float)
    F["brk60"] = (C > hh[60].shift(1)).astype(float)
    F["brk250"] = (C > hh[250].shift(1)).astype(float)
    F["newlow20"] = (C < ll[20].shift(1)).astype(float)
    span60 = (hh[60] - ll[60]).replace(0, np.nan)
    F["pos60"] = (C - ll[60]) / span60
    # 피보나치 되돌림: 최근 60일 안에서 저점이 먼저, 고점이 나중에 나온 상승 파동(30% 이상)의 되돌림 비율
    ds_high = pd.Series(_days_since(H.to_numpy(), 60, "max"))
    ds_low = pd.Series(_days_since(L.to_numpy(), 60, "min"))
    upswing = (ds_low > ds_high) & (hh[60] / ll[60] - 1 >= 0.3)
    retr = ((hh[60] - C) / span60).where(upswing)
    F["fib_retr"] = retr
    for name, lvl in (("fib382", 0.382), ("fib50", 0.5), ("fib618", 0.618)):
        F[name] = ((retr - lvl).abs() < 0.04).astype(float)
    F["fib_zone"] = ((retr >= 0.342) & (retr <= 0.658)).astype(float)
    F["days_since_high60"] = ds_high

    # ---------- 캔들 ----------
    r = rng.replace(0, np.nan)
    lower = (np.minimum(O, C) - L) / r
    F["lower_wick"] = lower
    F["upper_wick"] = (H - np.maximum(O, C)) / r
    F["body"] = (C - O) / r
    F["close_loc"] = (C - L) / r
    F["hammer"] = ((lower >= 0.6) & (C / ll[20] - 1 <= 0.03)).astype(float)  # 저점 부근 긴 아래꼬리
    F["engulf"] = ((C > O) & (pc < O.shift(1)) & (C >= O.shift(1)) & (O <= pc)).astype(float)  # 상승 장악형
    F["gap"] = O / pc - 1
    F["up_streak"] = _streak(C > pc)
    F["down_streak"] = _streak(C < pc)

    # ---------- BTC 대비 ----------
    B = pd.Series(btc_c, dtype="float64")
    rb = B / B.shift(1) - 1
    for k in (7, 14, 28):
        F[f"rs{k}"] = roc[k] - (B / B.shift(k) - 1)
    ratio = C / B
    F["d_ratio50"] = ratio / ratio.rolling(50).mean() - 1
    F["ratio_high60"] = (ratio >= ratio.rolling(60).max()).astype(float)
    cov = ret1.rolling(60).cov(rb)
    F["beta60"] = cov / rb.rolling(60).var().replace(0, np.nan)
    F["corr60"] = ret1.rolling(60).corr(rb)

    # ---------- 급등 이력·기타 ----------
    pump = (ret1 >= 0.2).astype(float)
    F["pumps90"] = pump.rolling(90, min_periods=30).sum()
    last = np.where(pump.to_numpy() > 0, np.arange(n), -1)
    last = np.maximum.accumulate(last)
    F["days_since_pump"] = pd.Series(np.where(last >= 0, np.minimum(np.arange(n) - last, 365), 365).astype(float))
    F["max30"] = ret1.rolling(30).max()
    F["runup60"] = hh[60] / ll[60] - 1
    F["age"] = pd.Series(np.minimum(np.arange(n), 1000).astype(float))
    F["price_log"] = np.log10(C)

    # ---------- 수급 (김치프리미엄·펀딩비) ----------
    for key, win, minp in (("prem", 5, 3), ("fund", 7, 4)):
        x = (extra or {}).get(key)
        F[f"{key}{win}"] = pd.Series(x, dtype="float64").rolling(win, min_periods=minp).mean() if x is not None else pd.Series(np.full(n, np.nan))

    out = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan).astype("float32")
    out["val20"] = val20.to_numpy()
    out["ret1"] = ret1.to_numpy()
    return out


def targets(o, h, l, c, cost, horizon=7):
    """그날 종가에 신호를 보고 다음 날 시가에 샀을 때의 결과 (왕복 비용 반영)."""
    n = len(c)
    out = {k: np.full(n, np.nan) for k in ("entry", "fwd", "mx", "mn", "tp30sl10", "tp50sl15", "sl10", "tp20sl7")}
    if n <= horizon + 1:
        return pd.DataFrame(out)
    m = n - horizon - 1  # 신호일 0..m 까지 계산 가능
    entry = o[1 : m + 2]
    Wh = swv(h[1:], horizon)[: m + 1]
    Wl = swv(l[1:], horizon)[: m + 1]
    Wo = swv(o[1:], horizon)[: m + 1]
    last_c = c[horizon : horizon + m + 1]
    out["entry"][: m + 1] = entry
    out["fwd"][: m + 1] = last_c / entry * (1 - cost) / (1 + cost) - 1
    out["mx"][: m + 1] = Wh.max(axis=1) / entry - 1
    out["mn"][: m + 1] = Wl.min(axis=1) / entry - 1

    def path(tp, sl):
        tgt = entry * (1 + tp) if tp else np.full(m + 1, np.inf)
        stop = entry * (1 - sl) if sl else np.zeros(m + 1)
        hit_s = Wl <= stop[:, None]
        hit_t = Wh >= tgt[:, None]
        fs = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1), 99)
        ft = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), 99)
        rows = np.arange(m + 1)
        px = last_c.copy()
        s_first = (fs <= ft) & (fs < 99)  # 같은 날 둘 다 닿으면 손절이 먼저라고 본다 (보수적)
        t_first = (ft < fs) & (ft < 99)
        px[s_first] = np.minimum(Wo[rows[s_first], fs[s_first]], stop[s_first])
        px[t_first] = np.maximum(Wo[rows[t_first], ft[t_first]], tgt[t_first])
        return px / entry * (1 - cost) / (1 + cost) - 1

    out["tp30sl10"][: m + 1] = path(0.30, 0.10)
    out["tp50sl15"][: m + 1] = path(0.50, 0.15)
    out["sl10"][: m + 1] = path(0, 0.10)
    out["tp20sl7"][: m + 1] = path(0.20, 0.07)
    return pd.DataFrame(out)


# 보고서용 이름
LABEL = {
    "d_sma5": "5일선 대비 위치", "d_sma10": "10일선 대비 위치", "d_sma20": "20일선 대비 위치", "d_sma50": "50일선 대비 위치",
    "d_sma100": "100일선 대비 위치", "d_sma200": "200일선 대비 위치", "slope_sma20": "20일선 기울기", "slope_sma50": "50일선 기울기",
    "ma_align": "이평선 정배열", "ma_align_dn": "이평선 역배열", "gc_5_20": "골든크로스 5·20일 (최근 3일)", "dc_5_20": "데드크로스 5·20일",
    "gc_20_60": "골든크로스 20·60일", "dc_20_60": "데드크로스 20·60일", "gc_50_200": "골든크로스 50·200일", "dc_50_200": "데드크로스 50·200일",
    "macd": "MACD 값", "macd_hist": "MACD 히스토그램", "macd_cross_up": "MACD 골든크로스", "macd_cross_up_neg": "MACD 골든크로스 (0선 아래)",
    "adx14": "ADX (추세 강도)", "di_diff": "+DI − −DI", "ichi_cloud": "일목 구름 위/아래", "ichi_d_cloud": "일목 구름 상단 대비 위치",
    "ichi_tk": "일목 전환선/기준선", "ichi_tk_cross": "일목 전환선 상향 교차", "ichi_break_up": "일목 구름 상향 돌파",
    "rsi14": "RSI(14)", "rsi2": "RSI(2)", "rsi_cross30": "RSI 30 상향 돌파", "bull_div": "RSI 상승 다이버전스",
    "stoch_k": "스토캐스틱 %K", "stoch_kd": "스토캐스틱 %K−%D", "stoch_cross_os": "스토캐스틱 과매도권 골든크로스", "willr14": "윌리엄스 %R",
    "cci20": "CCI(20)", "mfi14": "MFI(14)", "roc1": "1일 등락률", "roc3": "3일 등락률", "roc7": "7일 등락률", "roc14": "14일 등락률",
    "roc28": "28일 등락률", "roc60": "60일 등락률", "roc120": "120일 등락률", "atr_pct": "변동성 (ATR%)", "atr_ratio": "변동성 확대 (5일/60일)",
    "bb_pctb": "볼린저 %B", "bb_width": "볼린저 밴드폭", "bb_width_pct": "볼린저 밴드폭 백분위 (낮으면 스퀴즈)", "kc_squeeze": "켈트너 스퀴즈",
    "nr7": "NR7 (7일 중 최소 변동폭)", "range10": "10일 가격 범위", "rv_pct": "변동성 백분위 (1년)", "vol_ratio": "거래대금 배수 (오늘/20일)",
    "vol5_60": "거래대금 증가 (5일/60일)", "vol10_30": "거래대금 증가 (10일/30일)", "obv_slope20": "OBV 기울기", "updown_vol20": "상승일/하락일 거래량 비",
    "val20_log": "거래대금 규모", "accum": "조용한 매집 (가격 횡보 + 거래량 증가)", "vol_dry": "거래량 급감", "d_high20": "20일 고점 대비",
    "d_high60": "60일 고점 대비", "d_high120": "120일 고점 대비", "d_high250": "250일 고점 대비", "d_low20": "20일 저점 대비",
    "d_low60": "60일 저점 대비", "d_low250": "250일 저점 대비", "brk20": "20일 고점 돌파", "brk60": "60일 고점 돌파", "brk250": "250일 고점 돌파",
    "newlow20": "20일 저점 이탈", "pos60": "60일 범위 내 위치", "fib_retr": "피보나치 되돌림 비율", "fib382": "피보나치 0.382 되돌림",
    "fib50": "피보나치 0.5 되돌림", "fib618": "피보나치 0.618 되돌림", "fib_zone": "피보나치 0.382~0.618 구간", "days_since_high60": "60일 고점 이후 경과일",
    "lower_wick": "아래꼬리 비율", "upper_wick": "위꼬리 비율", "body": "몸통 방향·크기", "close_loc": "종가의 당일 범위 내 위치", "hammer": "망치형 (저점 부근 긴 아래꼬리)",
    "engulf": "상승 장악형", "gap": "시가 갭", "up_streak": "연속 상승일", "down_streak": "연속 하락일", "rs7": "BTC 대비 7일 강도", "rs14": "BTC 대비 14일 강도",
    "rs28": "BTC 대비 28일 강도", "d_ratio50": "BTC 대비 비율의 50일선 위치", "ratio_high60": "BTC 대비 비율 60일 신고가", "beta60": "BTC 베타", "corr60": "BTC 상관계수",
    "pumps90": "최근 90일 급등 횟수", "days_since_pump": "마지막 급등 이후 경과일", "max30": "30일 최대 일간 상승률", "runup60": "60일 저점→고점 폭",
    "age": "상장 후 경과일", "price_log": "가격 단위 (동전주 여부)", "prem5": "김치프리미엄", "fund7": "펀딩비 7일 평균",
}
BINARY = [
    "ma_align", "ma_align_dn", "gc_5_20", "dc_5_20", "gc_20_60", "dc_20_60", "gc_50_200", "dc_50_200", "macd_cross_up", "macd_cross_up_neg",
    "ichi_tk_cross", "ichi_break_up", "rsi_cross30", "bull_div", "stoch_cross_os", "kc_squeeze", "nr7", "accum", "vol_dry", "brk20", "brk60", "brk250",
    "newlow20", "fib382", "fib50", "fib618", "fib_zone", "hammer", "engulf", "ratio_high60",
]
# 연속형 지표를 문턱값으로 자른 신호 (조합 실험용)
THRESH = {
    "rsi_lt30": ("rsi14", "<", 30, "RSI 30 미만"), "rsi_gt70": ("rsi14", ">", 70, "RSI 70 초과"), "above_sma50": ("d_sma50", ">", 0, "50일선 위"),
    "above_sma200": ("d_sma200", ">", 0, "200일선 위"), "cloud_above": ("ichi_cloud", ">", 0.5, "일목 구름 위"), "squeeze": ("bb_width_pct", "<", 0.1, "볼린저 스퀴즈 (밴드폭 하위 10%)"),
    "bb_low": ("bb_pctb", "<", 0, "볼린저 하단 아래"), "bb_high": ("bb_pctb", ">", 1, "볼린저 상단 위"), "vol_up": ("vol5_60", ">", 2, "거래대금 5일 평균이 60일의 2배 이상"),
    "vol_spike": ("vol_ratio", ">", 3, "오늘 거래대금 3배 이상"), "near_high250": ("d_high250", ">", -0.1, "250일 고점 10% 이내"), "near_low60": ("d_low60", "<", 0.05, "60일 저점 5% 이내"),
    "pumped_before": ("pumps90", ">", 0.5, "최근 90일 안에 급등한 적 있음"), "no_pump_year": ("days_since_pump", ">", 364, "1년간 급등 없음"), "strong_vs_btc": ("rs28", ">", 0.2, "BTC보다 28일간 20%p 이상 강함"),
    "tight_range": ("range10", "<", 0.12, "10일 범위 12% 미만 (횡보)"), "adx_trend": ("adx14", ">", 30, "ADX 30 초과"), "down_streak3": ("down_streak", ">", 2.5, "3일 이상 연속 하락"),
    "low_price": ("price_log", "<", 2, "가격 단위 100 미만"), "premium_high": ("prem5", ">", 0.03, "김치프리미엄 3% 초과"), "funding_neg": ("fund7", "<", 0, "펀딩비 음수"),
}
