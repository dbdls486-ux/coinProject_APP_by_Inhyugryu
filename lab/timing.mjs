/**
 * 메이저 코인 한 종목을 규칙대로 사고파는 타이밍 실험.
 * 워뇨띠가 밝힌 방식("편승하지 않고 너무 싸면 매수, 너무 비싸면 매도", 박스권 매매, 5% 안팎 손절)을
 * 일봉 규칙으로 옮겨, 그냥 보유·50일선 추세 타이밍과 비교한다.
 *
 * 판단은 종가에, 체결은 다음 날 시가에 한다. 손절은 장중 저가가 닿으면 그 가격(갭이면 시가)에.
 */
import { rsi, sma, stdev } from "./indicators.mjs";

function rolling(x, n, fn) {
  const out = new Float64Array(x.length).fill(NaN);
  for (let i = n - 1; i < x.length; i++) {
    let v = x[i];
    for (let j = i - n + 1; j < i; j++) v = fn(v, x[j]);
    out[i] = v;
  }
  return out;
}

function context(k) {
  const c = k.c;
  const s20 = sma(c, 20);
  const sd20 = stdev(c, 20);
  return {
    k,
    c,
    rsi14: rsi(c, 14),
    rsi2: rsi(c, 2),
    sma5: sma(c, 5),
    sma20: s20,
    sma50: sma(c, 50),
    sma200: sma(c, 200),
    bbLow: Float64Array.from(s20, (m, i) => m - 2 * sd20[i]),
    hi30: rolling(k.h, 30, Math.max),
    lo30: rolling(k.l, 30, Math.min),
  };
}

export const TIMING_RULES = [
  { id: "HOLD", label: "그냥 보유", enter: () => true, exit: () => false },
  { id: "TREND50", label: "50일선 위에서만 보유 (추세)", enter: (x, i) => x.c[i] > x.sma50[i], exit: (x, i) => x.c[i] < x.sma50[i] },
  { id: "DIP_RSI", label: "RSI 30 아래에서 매수 → 55 위 또는 20일 뒤 매도", enter: (x, i) => x.rsi14[i] < 30, exit: (x, i, held) => x.rsi14[i] > 55 || held >= 20 },
  { id: "DIP_BB", label: "볼린저 하단 아래에서 매수 → 20일선 회복 또는 20일 뒤 매도", enter: (x, i) => x.c[i] < x.bbLow[i], exit: (x, i, held) => x.c[i] > x.sma20[i] || held >= 20 },
  {
    id: "RANGE30",
    label: "30일 박스 하단 20%에서 매수 → 상단 20% 또는 30일 뒤 매도",
    enter: (x, i) => (x.c[i] - x.lo30[i]) / (x.hi30[i] - x.lo30[i]) < 0.2,
    exit: (x, i, held) => (x.c[i] - x.lo30[i]) / (x.hi30[i] - x.lo30[i]) > 0.8 || held >= 30,
  },
  { id: "DIP_UPTREND", label: "200일선 위에서 RSI(2) 10 아래 눌림 매수 → 5일선 회복 또는 10일 뒤 매도", enter: (x, i) => x.c[i] > x.sma200[i] && x.rsi2[i] < 10, exit: (x, i, held) => x.c[i] > x.sma5[i] || held >= 10 },
  { id: "DROP3D", label: "3일간 8% 넘게 빠지면 매수 → +5% 또는 10일 뒤 매도", enter: (x, i) => i >= 3 && x.c[i] / x.c[i - 3] - 1 <= -0.08, exit: (x, i, held, entry) => x.c[i] >= entry * 1.05 || held >= 10 },
];

/** @param stopPct 주어지면 매수가 대비 그만큼 빠질 때 손절 */
export function ruleTiming(U, k, a, b, fee, rule, stopPct = 0) {
  const x = context(k);
  let cash = 1;
  let qty = 0;
  let entryPx = 0;
  let entryI = 0;
  let entryCash = 1;
  let buy = false;
  let sell = false;
  let peak = 1;
  let mdd = 0;
  let prev = 1;
  let inDays = 0;
  const rs = [];
  const tr = [];
  const close = (px) => {
    cash = qty * px * (1 - fee);
    tr.push(cash / entryCash - 1);
    qty = 0;
  };
  for (let t = a; t <= b; t++) {
    const i = k.idx[t];
    if (i < 0) continue;
    if (buy && qty === 0) {
      entryCash = cash;
      qty = (cash * (1 - fee)) / k.o[i];
      cash = 0;
      entryPx = k.o[i];
      entryI = i;
    } else if (sell && qty > 0) close(k.o[i]);
    buy = sell = false;
    if (qty > 0 && stopPct > 0 && k.l[i] <= entryPx * (1 - stopPct)) close(Math.min(k.o[i], entryPx * (1 - stopPct)));
    const eq = cash + qty * k.c[i];
    if (qty > 0) inDays++;
    peak = Math.max(peak, eq);
    mdd = Math.min(mdd, eq / peak - 1);
    rs.push(eq / prev - 1);
    prev = eq;
    if (qty > 0) sell = !!rule.exit(x, i, i - entryI, entryPx);
    else buy = !!rule.enter(x, i);
  }
  if (rs.length < 30) return null;
  if (qty > 0) tr.push((qty * k.c[k.idx[b] >= 0 ? k.idx[b] : k.n - 1] * (1 - fee)) / entryCash - 1);
  const days = b - a + 1;
  const total = prev - 1;
  const mean = rs.reduce((s, v) => s + v, 0) / rs.length;
  const sd = Math.sqrt(rs.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, rs.length - 1));
  return {
    cagr: total > -1 ? (1 + total) ** (365 / days) - 1 : -1,
    mdd,
    sharpe: sd > 0 ? (mean / sd) * Math.sqrt(365) : 0,
    exposure: inDays / rs.length,
    trades: tr.length,
    winRate: tr.length ? tr.filter((v) => v > 0).length / tr.length : 0,
    avgTrade: tr.length ? tr.reduce((s, v) => s + v, 0) / tr.length : 0,
  };
}

export const MAJORS = ["BTC", "ETH", "XRP", "SOL", "DOGE"];

/** 메이저 코인 × 규칙 × (손절 없음 / 5% 손절) */
export function majorsTiming(U, P, fee) {
  const out = [];
  for (const sym of MAJORS) {
    const k = U.coins.find((c) => c.sym === sym);
    if (!k) continue;
    for (const rule of TIMING_RULES) {
      for (const stop of rule.id === "HOLD" || rule.id === "TREND50" ? [0] : [0, 0.05]) {
        out.push({ sym, rule: rule.id, label: rule.label, stop, train: ruleTiming(U, k, ...P.train, fee, rule, stop), test: ruleTiming(U, k, ...P.test, fee, rule, stop) });
      }
    }
  }
  return out;
}
