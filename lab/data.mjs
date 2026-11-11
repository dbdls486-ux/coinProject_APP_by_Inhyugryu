/**
 * 수집한 일봉을 읽어 날짜 축을 맞추고, 종목별 지표와 날짜별 순위(백분위)를 미리 계산한다.
 * 모든 값은 그날 종가까지의 정보로만 만든다.
 */
import { readdir, readFile } from "node:fs/promises";
import { atr, highestPrev, maxDailyRet, ret, rsi, sma, smaPrev, stdev } from "./indicators.mjs";

/** 스테이블코인·원화 연동 자산 — 매매 대상에서 뺀다 */
const STABLE = new Set(["USDT", "USDC", "USDS", "USDE", "DAI", "USD1", "PYUSD", "FDUSD", "TUSD", "BUSD", "RLUSD", "USDP", "GUSD", "EURC", "USDQ", "USDG"]);

export const MIN_AGE = 60; // 상장 후 최소 일수 (지표가 안정되기 전엔 거래하지 않는다)

/**
 * @param {string} dir lab/.data
 * @param {{ exchange: string, minValue: number }} opts minValue: 최근 20일 평균 거래대금 하한 (그 거래소 호가 통화)
 */
export async function loadUniverse(dir, opts) {
  const sub = `${dir}/${opts.exchange}`;
  const files = (await readdir(sub)).filter((f) => f.endsWith(".json"));
  const raw = [];
  let truncated = 0;
  for (const f of files) {
    const j = JSON.parse(await readFile(`${sub}/${f}`, "utf8"));
    // 아직 끝나지 않은 오늘 봉은 뺀다 (종가가 확정되지 않았다)
    if (opts.dropFrom) j.rows = j.rows.filter((r) => r[0] < opts.dropFrom);
    // 날짜 중복·역순 방지
    for (let i = 1; i < j.rows.length; i++) if (!(j.rows[i][0] > j.rows[i - 1][0])) throw new Error(`${opts.exchange}/${j.sym}: 날짜가 정렬돼 있지 않거나 중복 (${j.rows[i][0]})`);
    // 하루에 20배 넘게 뛰거나 95% 넘게 빠진 봉은 대부분 액면 변경·토큰 교체다. 그 이전 기록은 다른 코인으로 보고 버린다
    let cut = 0;
    for (let i = 1; i < j.rows.length; i++) {
      const r = Number(j.rows[i][4]) / Number(j.rows[i - 1][4]);
      if (r > 20 || r < 0.05) cut = i;
    }
    if (cut > 0) {
      j.rows = j.rows.slice(cut);
      truncated++;
    }
    if (j.rows.length >= 30) raw.push(j);
  }
  // 날짜 축: 모든 종목 날짜의 합집합
  const dates = [...new Set(raw.flatMap((j) => j.rows.map((r) => r[0])))].sort();
  const dateIdx = new Map(dates.map((d, i) => [d, i]));
  const T = dates.length;

  const coins = raw.map((j) => {
    const n = j.rows.length;
    const col = (k) => Float64Array.from(j.rows, (r) => Number(r[k]));
    const o = col(1), h = col(2), l = col(3), c = col(4), v = col(5);
    const idx = new Int32Array(T).fill(-1); // 날짜 → 이 종목의 몇 번째 봉인지
    const day = new Int32Array(n); // 이 종목의 봉 → 날짜 번호
    j.rows.forEach((r, i) => {
      const t = dateIdx.get(r[0]);
      idx[t] = i;
      day[i] = t;
    });
    const s20 = sma(c, 20);
    const sd20 = stdev(c, 20);
    const f = {
      sma20: s20,
      sma50: sma(c, 50),
      sma100: sma(c, 100),
      rsi14: rsi(c, 14),
      rsi2: rsi(c, 2),
      atr14: atr(h, l, c, 14),
      ret7: ret(c, 7),
      ret14: ret(c, 14),
      ret28: ret(c, 28),
      hh10: highestPrev(h, 10),
      hh20: highestPrev(h, 20),
      hh40: highestPrev(h, 40),
      hh60: highestPrev(h, 60),
      val20: sma(v, 20),
      volRatio: (() => {
        const p = smaPrev(v, 20);
        return Float64Array.from(v, (x, i) => (p[i] > 0 ? x / p[i] : NaN));
      })(),
      max30: maxDailyRet(c, 30),
      bbLow: Float64Array.from(s20, (m, i) => m - 2 * sd20[i]),
    };
    f.atrPct = Float64Array.from(f.atr14, (a, i) => a / c[i]);
    const stable = STABLE.has(j.sym);
    const eligible = new Uint8Array(n);
    for (let i = 0; i < n; i++) eligible[i] = !stable && i >= MIN_AGE && f.val20[i] >= opts.minValue ? 1 : 0;
    // bucket: 그날 거래 가능한 종목 사이에서 변동성(ATR%) 5분위 (0~4). 무작위 비교를 같은 변동성끼리 하려고
    return { sym: j.sym, name: j.name, n, o, h, l, c, v, idx, day, f, eligible, p: {}, bucket: new Int8Array(n).fill(-1) };
  });

  const btc = coins.find((k) => k.sym === "BTC");
  if (!btc) throw new Error("BTC 데이터가 없습니다");

  // 날짜별 BTC 상태 (시장 국면 필터)
  const btcUp = new Uint8Array(T);
  const btcRet14 = new Float64Array(T).fill(NaN);
  for (let t = 0; t < T; t++) {
    const i = btc.idx[t];
    if (i < 0) continue;
    btcUp[t] = btc.c[i] > btc.f.sma50[i] ? 1 : 0;
    btcRet14[t] = btc.f.ret14[i];
  }
  // BTC 대비 14일 초과 수익
  for (const k of coins) k.f.rs14 = Float64Array.from(k.f.ret14, (r, i) => r - btcRet14[k.day[i]]);

  // 날짜별 백분위 (그날 거래 가능한 종목끼리 비교)
  const RANKED = ["ret7", "ret14", "ret28", "rs14", "max30"];
  for (const name of RANKED) for (const k of coins) k.p[name] = new Float64Array(k.n).fill(NaN);
  const universeSize = new Int32Array(T);
  for (let t = 0; t < T; t++) {
    const live = [];
    for (const k of coins) {
      const i = k.idx[t];
      if (i >= 0 && k.eligible[i]) live.push([k, i]);
    }
    universeSize[t] = live.length;
    if (live.length < 10) continue;
    const byVol = live.filter(([k, i]) => Number.isFinite(k.f.atrPct[i])).sort((a, b) => a[0].f.atrPct[a[1]] - b[0].f.atrPct[b[1]]);
    byVol.forEach(([k, i], r) => (k.bucket[i] = Math.min(4, Math.floor((r * 5) / byVol.length))));
    for (const name of RANKED) {
      const vals = live.filter(([k, i]) => Number.isFinite(k.f[name][i])).sort((a, b) => a[0].f[name][a[1]] - b[0].f[name][b[1]]);
      const m = vals.length;
      vals.forEach(([k, i], r) => (k.p[name][i] = m > 1 ? r / (m - 1) : 0.5));
    }
  }

  return { dates, T, coins, btc, btcUp, universeSize, truncated };
}
