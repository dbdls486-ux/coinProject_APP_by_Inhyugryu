/**
 * 가격 차트 밖의 지표 — 김치프리미엄, 선물 펀딩비.
 * 두 거래소의 일봉 마감 시각이 달라서 미래 정보가 섞이지 않게 날짜를 맞춘다.
 *   빗썸 일봉 마감: 한국 자정 = UTC 15:00 (같은 날짜)
 *   바이낸스 일봉 마감: UTC 자정 = 다음 날 한국 09:00
 *   펀딩비: UTC 00·08·16시 정산 → 날짜 D 의 평균은 D 다음 날 UTC 00:00 에 확정
 */
import { readFile } from "node:fs/promises";

const prevDate = (d) => new Date(Date.parse(d + "T00:00:00Z") - 86_400_000).toISOString().slice(0, 10);

/** 날짜 문자열 → 종가 */
function closeMap(k, U) {
  const m = new Map();
  for (let i = 0; i < k.n; i++) m.set(U.dates[k.day[i]], k.c[i]);
  return m;
}

function rollingMean(vals, n) {
  const out = new Float64Array(vals.length).fill(NaN);
  for (let i = 0; i < vals.length; i++) {
    let s = 0;
    let c = 0;
    for (let j = Math.max(0, i - n + 1); j <= i; j++) {
      if (Number.isFinite(vals[j])) {
        s += vals[j];
        c++;
      }
    }
    if (c >= Math.ceil(n / 2)) out[i] = s / c;
  }
  return out;
}

/** 그날 거래 가능한 종목끼리 백분위 */
function rankDaily(U, name) {
  for (const k of U.coins) k.p[name] = new Float64Array(k.n).fill(NaN);
  for (let t = 0; t < U.T; t++) {
    const vals = U.live[t].filter(([k, i]) => Number.isFinite(k.f[name][i])).sort((a, b) => a[0].f[name][a[1]] - b[0].f[name][b[1]]);
    const m = vals.length;
    if (m < 5) continue;
    vals.forEach(([k, i], r) => (k.p[name][i] = r / (m - 1)));
  }
}

/**
 * 김치프리미엄 (5일 평균). 두 시장 모두에 붙인다.
 *   빗썸 신호(한국 자정 마감): 빗썸 종가 / (바이낸스 전날 종가 × 빗썸 USDT 종가) − 1
 *   바이낸스 신호(UTC 자정 마감): 빗썸 같은 날 종가 / (바이낸스 같은 날 종가 × 빗썸 USDT 같은 날 종가) − 1
 * 시장 전체 프리미엄(환율·USDT 프리미엄)은 날짜별 백분위로 비교하므로 상쇄된다.
 */
export function attachKimchi(Ub, Un) {
  const usdt = Ub.coins.find((k) => k.sym === "USDT");
  if (!usdt) return false;
  const usdtC = closeMap(usdt, Ub);
  const bnBy = new Map(Un.coins.map((k) => [k.sym, k]));
  const bhBy = new Map(Ub.coins.map((k) => [k.sym, k]));
  const bnClose = new Map();
  const bhClose = new Map();
  for (const k of Ub.coins) {
    const kb = bnBy.get(k.sym);
    const prem = new Float64Array(k.n).fill(NaN);
    if (kb) {
      const m = bnClose.get(k.sym) ?? closeMap(kb, Un);
      bnClose.set(k.sym, m);
      for (let i = 0; i < k.n; i++) {
        const d = Ub.dates[k.day[i]];
        const p = m.get(prevDate(d));
        const u = usdtC.get(d);
        if (p > 0 && u > 0) prem[i] = k.c[i] / (p * u) - 1;
      }
    }
    k.f.prem5 = rollingMean(prem, 5);
  }
  for (const k of Un.coins) {
    const kh = bhBy.get(k.sym);
    const prem = new Float64Array(k.n).fill(NaN);
    if (kh) {
      const m = bhClose.get(k.sym) ?? closeMap(kh, Ub);
      bhClose.set(k.sym, m);
      for (let i = 0; i < k.n; i++) {
        const d = Un.dates[k.day[i]];
        const p = m.get(d);
        const u = usdtC.get(d);
        if (p > 0 && u > 0) prem[i] = p / (k.c[i] * u) - 1;
      }
    }
    k.f.prem5 = rollingMean(prem, 5);
  }
  rankDaily(Ub, "prem5");
  rankDaily(Un, "prem5");
  return true;
}

/**
 * 펀딩비 7일 평균. 바이낸스 신호는 같은 날까지, 빗썸 신호는 전날까지의 펀딩비를 쓴다
 * (빗썸 마감 시각에는 그날 마지막 정산이 아직 안 끝났다).
 */
export async function attachFunding(U, dir, lagDays) {
  let found = 0;
  for (const k of U.coins) {
    const fr = new Float64Array(k.n).fill(NaN);
    try {
      const j = JSON.parse(await readFile(`${dir}/funding/${k.sym}.json`, "utf8"));
      const m = new Map(j.rows);
      for (let i = 0; i < k.n; i++) {
        let d = U.dates[k.day[i]];
        for (let l = 0; l < lagDays; l++) d = prevDate(d);
        const v = m.get(d);
        if (Number.isFinite(v)) fr[i] = v;
      }
      found++;
    } catch {
      /* 선물이 없는 종목 */
    }
    k.f.fund7 = rollingMean(fr, 7);
  }
  rankDaily(U, "fund7");
  return found;
}
