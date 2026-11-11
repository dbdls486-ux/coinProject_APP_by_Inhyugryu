/**
 * 비교 기준이자 지금까지 유일하게 통한 신호 — 메이저 코인 추세 타이밍.
 *  - 그냥 보유
 *  - N일선 타이밍: 종가가 N일선 위면 다음 날 시가에 사고, 아래로 내려가면 다음 날 시가에 판다 (수수료 반영)
 * 50일선이 우연히 맞은 숫자가 아닌지 20~200일선 전체로 확인한다.
 */
import { sma } from "./indicators.mjs";

export const TREND_LENGTHS = [20, 30, 50, 100, 150, 200];

function stats(rs, total, days, mdd, extra) {
  const mean = rs.reduce((s, x) => s + x, 0) / Math.max(1, rs.length);
  const sd = Math.sqrt(rs.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(1, rs.length - 1));
  return { days, total, cagr: total > -1 ? (1 + total) ** (365 / days) - 1 : -1, mdd, sharpe: sd > 0 ? (mean / sd) * Math.sqrt(365) : 0, ...extra };
}

/** 한 종목의 N일선 타이밍 (n 이 없으면 그냥 보유) */
export function trendTiming(U, k, a, b, fee, n) {
  const ma = n ? sma(k.c, n) : null;
  let cash = 1;
  let qty = 0;
  let wantIn = !n; // 그냥 보유는 첫날 산다
  let peak = 1;
  let mdd = 0;
  let trades = 0;
  let inDays = 0;
  let prev = 1;
  let started = false;
  const rs = [];
  for (let t = a; t <= b; t++) {
    const i = k.idx[t];
    if (i < 0) continue;
    if (wantIn && qty === 0) {
      qty = (cash * (1 - fee)) / k.o[i];
      cash = 0;
      trades++;
    } else if (!wantIn && qty > 0) {
      cash = qty * k.o[i] * (1 - fee);
      qty = 0;
    }
    const eq = cash + qty * k.c[i];
    if (qty > 0) inDays++;
    peak = Math.max(peak, eq);
    mdd = Math.min(mdd, eq / peak - 1);
    rs.push(eq / prev - 1);
    prev = eq;
    started = true;
    if (n) wantIn = Number.isFinite(ma[i]) && k.c[i] > ma[i];
  }
  if (!started) return null;
  return stats(rs, prev - 1, b - a + 1, mdd, { trades, exposure: inDays / Math.max(1, rs.length) });
}

/** 기존 보고서 호환: BTC 50일선 */
export function btcTiming(U, a, b, fee) {
  return trendTiming(U, U.btc, a, b, fee, 50);
}

/** BTC·ETH 각각 그냥 보유와 여러 길이의 이평선 타이밍 */
export function trendGrid(U, P, fee) {
  const out = [];
  for (const sym of ["BTC", "ETH"]) {
    const k = U.coins.find((c) => c.sym === sym);
    if (!k) continue;
    for (const n of [0, ...TREND_LENGTHS]) {
      out.push({ sym, n, train: trendTiming(U, k, ...P.train, fee, n), test: trendTiming(U, k, ...P.test, fee, n) });
    }
  }
  return out;
}
