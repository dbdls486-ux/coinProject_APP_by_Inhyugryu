/**
 * 비교 기준 — 알트 스윙 규칙이 최소한 이것들보다는 나아야 쓸 의미가 있다.
 *  - BTC 그냥 보유
 *  - BTC 50일선 타이밍: 종가가 50일선 위면 다음 날 시가에 사고, 아래로 내려가면 다음 날 시가에 판다 (수수료 반영)
 */

export function btcTiming(U, a, b, fee) {
  const k = U.btc;
  let cash = 1;
  let qty = 0;
  let wantIn = false;
  let peak = 1;
  let mdd = 0;
  let trades = 0;
  let lastClose = NaN;
  const rs = [];
  let prev = 1;
  for (let t = a; t <= b; t++) {
    const i = k.idx[t];
    if (i < 0) continue;
    // 시가: 어제 종가 판단대로 사거나 판다
    if (wantIn && qty === 0) {
      qty = (cash * (1 - fee)) / k.o[i];
      cash = 0;
      trades++;
    } else if (!wantIn && qty > 0) {
      cash = qty * k.o[i] * (1 - fee);
      qty = 0;
    }
    lastClose = k.c[i];
    const eq = cash + qty * lastClose;
    peak = Math.max(peak, eq);
    mdd = Math.min(mdd, eq / peak - 1);
    rs.push(eq / prev - 1);
    prev = eq;
    wantIn = Number.isFinite(k.f.sma50[i]) && k.c[i] > k.f.sma50[i];
  }
  const total = prev - 1;
  const days = b - a + 1;
  const mean = rs.reduce((s, x) => s + x, 0) / rs.length;
  const sd = Math.sqrt(rs.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(1, rs.length - 1));
  return { days, total, cagr: (1 + total) ** (365 / days) - 1, mdd, sharpe: sd > 0 ? (mean / sd) * Math.sqrt(365) : 0, trades };
}
