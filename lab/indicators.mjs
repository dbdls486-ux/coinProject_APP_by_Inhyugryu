/**
 * 지표 계산. 모든 함수는 i 시점의 값을 i 시점까지의 데이터만으로 만든다 (미래 정보 없음).
 * 값이 정의되지 않는 앞부분은 NaN.
 */

const NaNArray = (n) => new Float64Array(n).fill(NaN);

export function sma(x, n) {
  const out = NaNArray(x.length);
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    s += x[i];
    if (i >= n) s -= x[i - n];
    if (i >= n - 1) out[i] = s / n;
  }
  return out;
}

/** 직전 n일 평균 (오늘 제외) — 오늘 거래대금이 평소의 몇 배인지 볼 때 */
export function smaPrev(x, n) {
  const m = sma(x, n);
  const out = NaNArray(x.length);
  for (let i = 1; i < x.length; i++) out[i] = m[i - 1];
  return out;
}

export function stdev(x, n) {
  const out = NaNArray(x.length);
  for (let i = n - 1; i < x.length; i++) {
    let s = 0;
    let s2 = 0;
    for (let j = i - n + 1; j <= i; j++) {
      s += x[j];
      s2 += x[j] * x[j];
    }
    const m = s / n;
    out[i] = Math.sqrt(Math.max(0, s2 / n - m * m));
  }
  return out;
}

/** Wilder RSI */
export function rsi(c, n) {
  const out = NaNArray(c.length);
  let gain = 0;
  let loss = 0;
  for (let i = 1; i < c.length; i++) {
    const d = c[i] - c[i - 1];
    const g = d > 0 ? d : 0;
    const l = d < 0 ? -d : 0;
    if (i <= n) {
      gain += g;
      loss += l;
      if (i === n) {
        gain /= n;
        loss /= n;
        out[i] = loss === 0 ? 100 : 100 - 100 / (1 + gain / loss);
      }
    } else {
      gain = (gain * (n - 1) + g) / n;
      loss = (loss * (n - 1) + l) / n;
      out[i] = loss === 0 ? 100 : 100 - 100 / (1 + gain / loss);
    }
  }
  return out;
}

/** Wilder ATR */
export function atr(h, l, c, n) {
  const out = NaNArray(c.length);
  let a = 0;
  for (let i = 1; i < c.length; i++) {
    const tr = Math.max(h[i] - l[i], Math.abs(h[i] - c[i - 1]), Math.abs(l[i] - c[i - 1]));
    if (i <= n) {
      a += tr;
      if (i === n) out[i] = a /= n;
    } else {
      a = (a * (n - 1) + tr) / n;
      out[i] = a;
    }
  }
  return out;
}

/** n일 수익률 */
export function ret(c, n) {
  const out = NaNArray(c.length);
  for (let i = n; i < c.length; i++) out[i] = c[i] / c[i - n] - 1;
  return out;
}

/** 직전 n일(오늘 제외) 최고가 — 돌파 판정용 */
export function highestPrev(h, n) {
  const out = NaNArray(h.length);
  for (let i = n; i < h.length; i++) {
    let m = -Infinity;
    for (let j = i - n; j < i; j++) if (h[j] > m) m = h[j];
    out[i] = m;
  }
  return out;
}

/** 최근 n일 중 가장 큰 하루 상승률 (MAX 효과) */
export function maxDailyRet(c, n) {
  const out = NaNArray(c.length);
  for (let i = n; i < c.length; i++) {
    let m = -Infinity;
    for (let j = i - n + 1; j <= i; j++) {
      const r = c[j] / c[j - 1] - 1;
      if (r > m) m = r;
    }
    out[i] = m;
  }
  return out;
}
