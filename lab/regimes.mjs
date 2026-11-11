/**
 * 시장 국면 필터 — "지금 알트코인을 사도 되는 시기인가".
 * 1차 검증에서 알트 스윙 규칙들이 2021년에는 크게 벌고 2022·2025년에는 크게 잃었다.
 * 신호보다 국면이 성적을 좌우하므로, 국면을 판단하는 방법 자체를 비교한다.
 *
 * 모든 값은 그날 종가까지의 정보로만 만든다. 결과는 날짜 번호(t)별 0/1.
 */

const sma = (x, n) => {
  const out = new Float64Array(x.length).fill(NaN);
  let s = 0;
  let cnt = 0;
  for (let i = 0; i < x.length; i++) {
    s += x[i];
    cnt++;
    if (cnt > n) {
      s -= x[i - n];
      cnt = n;
    }
    if (cnt === n) out[i] = s / n;
  }
  return out;
};

/**
 * @returns {{ id, label, on: Uint8Array }[]}
 */
export function buildRegimes(U) {
  const T = U.T;
  // 동일 비중 알트 지수: 그날 거래 가능한 BTC 외 종목의 하루 수익률 평균 (극단값은 ±50% 로 자른다)
  const altIdx = new Float64Array(T);
  const breadth = new Float64Array(T).fill(NaN);
  let level = 1;
  for (let t = 0; t < T; t++) {
    let sum = 0;
    let n = 0;
    let above = 0;
    let measured = 0;
    for (const [k, i] of U.live[t]) {
      if (k === U.btc) continue;
      if (i >= 1) {
        const r = Math.max(-0.5, Math.min(0.5, k.c[i] / k.c[i - 1] - 1));
        sum += r;
        n++;
      }
      const m = k.f.sma50[i];
      if (Number.isFinite(m)) {
        measured++;
        if (k.c[i] > m) above++;
      }
    }
    if (n > 0) level *= 1 + sum / n;
    altIdx[t] = level;
    if (measured >= 10) breadth[t] = above / measured;
  }
  const altSma = sma(altIdx, 50);

  // BTC 종가 (봉이 없는 날은 직전 값)
  const btcPx = new Float64Array(T);
  let last = NaN;
  for (let t = 0; t < T; t++) {
    const i = U.btc.idx[t];
    if (i >= 0) last = U.btc.c[i];
    btcPx[t] = last;
  }
  const ratio = Float64Array.from(altIdx, (a, t) => a / btcPx[t]);
  const ratioSma = sma(ratio, 50);

  const mk = (fn) => Uint8Array.from({ length: T }, (_, t) => (fn(t) ? 1 : 0));
  return [
    { id: "ALL", label: "항상", on: mk(() => true) },
    { id: "BTC50", label: "BTC가 50일선 위", on: U.btcUp },
    { id: "BREADTH", label: "알트 절반 이상이 50일선 위", on: mk((t) => breadth[t] >= 0.5) },
    { id: "ALT50", label: "알트 지수가 50일선 위", on: mk((t) => altIdx[t] > altSma[t]) },
    { id: "ALTBTC", label: "알트 지수가 BTC보다 강한 흐름 (비율이 50일선 위)", on: mk((t) => ratio[t] > ratioSma[t]) },
  ];
}
