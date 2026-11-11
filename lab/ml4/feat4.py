"""
4차 지표 — 계단식 하락과 그 뒤의 돌파를 잡기 위한 지표.
'어제까지'의 하락 상태(끝에 p가 붙은 것)와 '오늘'의 돌파를 나눠 계산한다. 모두 그날 종가까지의 정보만 쓴다.
"""
import numpy as np
import pandas as pd

swv = np.lib.stride_tricks.sliding_window_view
PIV = 3


def features4(o, h, l, c, v):
    n = len(c)
    O, H, L, C, V = (pd.Series(x) for x in (o, h, l, c, v))
    out = {}
    out["drop30p"] = (C.shift(1) / C.shift(31) - 1).to_numpy()
    out["drop60p"] = (C.shift(1) / C.shift(61) - 1).to_numpy()
    sma20, sma50 = C.rolling(20).mean(), C.rolling(50).mean()
    out["below20_30p"] = (C < sma20).astype(float).rolling(30).mean().shift(1).to_numpy()
    out["below50_60p"] = (C < sma50).astype(float).rolling(60).mean().shift(1).to_numpy()
    out["dd120p"] = (C.shift(1) / H.rolling(120).max().shift(1) - 1).to_numpy()
    out["reclaim50"] = ((C > sma50) & (C.shift(1) <= sma50.shift(1))).astype(float).to_numpy()
    out["vr10p"] = (V.rolling(10).mean().shift(1) / V.rolling(50).mean().shift(11).replace(0, np.nan)).to_numpy()  # 하락 중 거래량이 말랐는가
    out["day_body"] = (C / O - 1).to_numpy()
    # 확정된 고점·저점이 몇 번 연속으로 낮아졌는가 (계단식 하락)
    lh = np.zeros(n)
    ll = np.zeros(n)
    if n > 2 * PIV + 1:
        is_pl = np.zeros(n, bool)
        is_ph = np.zeros(n, bool)
        is_pl[PIV : n - PIV] = l[PIV : n - PIV] == swv(l, 2 * PIV + 1).min(axis=1)
        is_ph[PIV : n - PIV] = h[PIV : n - PIV] == swv(h, 2 * PIV + 1).max(axis=1)
        pl = ph = None
        rl = rh = 0
        for i in range(n):
            j = i - PIV
            if j >= 0:
                if is_pl[j]:
                    rl = rl + 1 if (pl is not None and l[j] < pl) else 0
                    pl = l[j]
                if is_ph[j]:
                    rh = rh + 1 if (ph is not None and h[j] < ph) else 0
                    ph = h[j]
            ll[i], lh[i] = rl, rh
    # 돌파 당일에는 이미 구조가 바뀌므로 어제 기준 값을 쓴다
    out["ll_run_p"] = np.r_[0.0, ll[:-1]]
    out["lh_run_p"] = np.r_[0.0, lh[:-1]]
    return pd.DataFrame({k: np.asarray(x, dtype="float32") for k, x in out.items()})


LABEL4 = {
    "drop30p": "돌파 전 30일 등락률", "drop60p": "돌파 전 60일 등락률", "below20_30p": "직전 30일 중 20일선 아래였던 비율", "below50_60p": "직전 60일 중 50일선 아래였던 비율",
    "dd120p": "120일 고점 대비 (어제)", "reclaim50": "50일선 회복", "vr10p": "직전 10일 거래량 (그 전 50일 대비)", "day_body": "당일 몸통 (종가/시가)", "ll_run_p": "저점 연속 낮춘 횟수", "lh_run_p": "고점 연속 낮춘 횟수",
    "mkt_brk": "그날 거래량 동반 60일 고점 돌파 종목 비율",
}
