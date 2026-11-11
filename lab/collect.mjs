#!/usr/bin/env node
/**
 * 검증 실험실 1단계 — 과거 시세 수집 (GitHub Actions 에서 실행)
 *
 *  - 빗썸 원화 마켓 전 종목의 일봉을 받을 수 있는 만큼 과거까지 받는다.
 *  - 바이낸스는 접속 가능 여부만 시험해 기록한다 (미국 서버에서 막히는지 확인용).
 *  - 결과는 lab/.data/ 에 저장한다. 원본 시세는 저장소에 커밋하지 않는다 (용량).
 *
 * 외부 패키지 없이 Node 22 내장 fetch 만 쓴다.
 */
import { mkdir, writeFile } from "node:fs/promises";

const OUT = process.env.LAB_DATA ?? "lab/.data";
const MAX_PAGES = Number(process.env.LAB_MAX_PAGES ?? 20); // 200일 × 20 = 약 11년
const WORKERS = 4;
const PAUSE_MS = 150;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getJson(url, tries = 6) {
  let last;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, { headers: { Accept: "application/json" } });
      if (res.status === 429 || res.status >= 500) {
        last = new Error(`HTTP ${res.status}`);
        await sleep(1000 * (i + 1));
        continue;
      }
      const text = await res.text();
      if (!res.ok) throw new Error(`HTTP ${res.status} ${text.slice(0, 200)}`);
      return JSON.parse(text);
    } catch (e) {
      last = e;
      await sleep(800 * (i + 1));
    }
  }
  throw last;
}

async function probe(url) {
  try {
    const res = await fetch(url, { headers: { Accept: "application/json" } });
    return { url, status: res.status };
  } catch (e) {
    return { url, status: 0, error: String(e?.cause?.code ?? e?.message ?? e) };
  }
}

/**
 * 과거 페이지를 부를 때 쓰는 to 형식. 빗썸 문서와 실제 동작이 다를 수 있어
 * BTC 로 여러 형식을 시험해 실제로 더 과거 봉이 나오는 형식을 고른다.
 */
const TO_FORMATS = {
  "utc-space": (k) => k.candle_date_time_utc.replace("T", " "),
  "kst-space": (k) => k.candle_date_time_kst.replace("T", " "),
  "utc-Z": (k) => `${k.candle_date_time_utc}Z`,
  "kst-offset": (k) => `${k.candle_date_time_kst}+09:00`,
  "utc-iso": (k) => k.candle_date_time_utc,
};
let toFormat = null;
const candleTimes = new Set(); // 일봉 시작 시각(한국 시각) — 한 가지여야 정상

/** 첫날~마지막 날 사이에 빠진 날 수 (페이지가 제대로 이어졌는지 확인) */
function missingDays(rows) {
  if (rows.length < 2) return 0;
  const span = Math.round((Date.parse(rows.at(-1)[0]) - Date.parse(rows[0][0])) / 86_400_000) + 1;
  return span - rows.length;
}

async function pickToFormat() {
  const first = await getJson("https://api.bithumb.com/v1/candles/days?market=KRW-BTC&count=200");
  const oldest = first.at(-1);
  let fallback = null;
  for (const [name, fmt] of Object.entries(TO_FORMATS)) {
    try {
      const q = new URLSearchParams({ market: "KRW-BTC", count: "200", to: fmt(oldest) });
      const list = await getJson(`https://api.bithumb.com/v1/candles/days?${q}`, 2);
      await sleep(PAUSE_MS);
      const newest = list?.[0]?.candle_date_time_utc;
      console.log(`  to 형식 ${name}: ${list?.length ?? 0}개, 최신 ${newest ?? "-"} (기준 ${oldest.candle_date_time_utc})`);
      const joined = newest && Date.parse(oldest.candle_date_time_utc + "Z") - Date.parse(newest + "Z") === 86_400_000;
      if (joined) return name;
      if (newest && newest < oldest.candle_date_time_utc) {
        console.log(`    → 과거로 넘어가긴 하지만 하루 차이로 이어지지 않음`);
        fallback ??= name;
      }
    } catch (e) {
      console.log(`  to 형식 ${name}: 실패 ${String(e?.message ?? e).slice(0, 120)}`);
    }
  }
  return fallback; // 이어지지 않는 형식이라도 쓰되, 종목별 빠진 날 수(meta)로 확인한다
}

/** 한 종목의 일봉을 최신 → 과거로 페이지를 넘기며 받는다. 반환은 오래된 순 */
async function bithumbDaily(market) {
  const rows = new Map();
  let to = "";
  for (let page = 0; page < (toFormat ? MAX_PAGES : 1); page++) {
    const q = new URLSearchParams({ market, count: "200" });
    if (to) q.set("to", to);
    const list = await getJson(`https://api.bithumb.com/v1/candles/days?${q}`);
    await sleep(PAUSE_MS);
    if (!Array.isArray(list) || list.length === 0) break;
    const before = rows.size;
    for (const k of list) {
      if (k.candle_date_time_kst) candleTimes.add(String(k.candle_date_time_kst).slice(11, 19));
      const d = String(k.candle_date_time_kst ?? k.candle_date_time_utc).slice(0, 10);
      // [날짜, 시가, 고가, 저가, 종가, 거래대금(원)]
      rows.set(d, [d, k.opening_price, k.high_price, k.low_price, k.trade_price, k.candle_acc_trade_price ?? 0]);
    }
    if (rows.size === before || list.length < 200) break; // 더 과거가 없거나 같은 페이지가 반복
    to = TO_FORMATS[toFormat](list.at(-1));
  }
  return [...rows.values()].sort((a, b) => (a[0] < b[0] ? -1 : 1));
}

/* ---------- 바이낸스 (data-api.binance.vision: 시세 전용 공개 주소, 미국 서버에서도 열림) ---------- */
const BN = "https://data-api.binance.vision";
const BN_START = Date.UTC(2017, 6, 1);
const STABLE_BASE = /^(USDC|FDUSD|TUSD|BUSD|DAI|USDP|USDE|USD1|PYUSD|EUR|EURI|AEUR|GBP|TRY|BRL|ARS|ZAR|UAH|RUB|NGN|IDRT|BIDR|PAXG|XUSD|RLUSD|USDS)$/;

async function binanceDaily(symbol) {
  const rows = [];
  let start = BN_START;
  for (let page = 0; page < 10; page++) {
    const q = new URLSearchParams({ symbol, interval: "1d", limit: "1000", startTime: String(start) });
    const list = await getJson(`${BN}/api/v3/klines?${q}`);
    await sleep(PAUSE_MS);
    if (!Array.isArray(list) || list.length === 0) break;
    // [날짜(UTC), 시가, 고가, 저가, 종가, 거래대금(USDT)]
    for (const k of list) rows.push([new Date(k[0]).toISOString().slice(0, 10), +k[1], +k[2], +k[3], +k[4], +k[7]]);
    if (list.length < 1000) break;
    start = list.at(-1)[0] + 86_400_000;
  }
  return rows;
}

async function collectBinance() {
  await mkdir(`${OUT}/binance`, { recursive: true });
  const info = await getJson(`${BN}/api/v3/exchangeInfo?permissions=SPOT`);
  const syms = info.symbols.filter(
    (s) => s.quoteAsset === "USDT" && s.status === "TRADING" && !STABLE_BASE.test(s.baseAsset),
  );
  console.log(`바이낸스 USDT 마켓 ${syms.length}개`);
  const queue = [...syms];
  const done = [];
  const failed = [];
  await Promise.all(
    Array.from({ length: WORKERS }, async () => {
      for (let s = queue.shift(); s; s = queue.shift()) {
        try {
          const rows = await binanceDaily(s.symbol);
          if (rows.length === 0) continue;
          await writeFile(`${OUT}/binance/${s.baseAsset}.json`, JSON.stringify({ sym: s.baseAsset, name: s.baseAsset, rows }));
          done.push({ sym: s.baseAsset, days: rows.length, first: rows[0][0], last: rows.at(-1)[0], missing: missingDays(rows) });
        } catch (e) {
          failed.push({ market: s.symbol, error: String(e?.message ?? e) });
        }
      }
    }),
  );
  const lens = done.map((d) => d.days).sort((a, b) => a - b);
  console.log(`바이낸스 완료: ${done.length}개, 실패 ${failed.length}개 · 봉 수 중앙값 ${lens[lens.length >> 1]}`);
  const gaps = done.filter((d) => d.missing > 0);
  return { coins: done.length, failed, coinsWithMissingDays: gaps.length, maxMissing: Math.max(0, ...gaps.map((d) => d.missing)) };
}

async function main() {
  await mkdir(`${OUT}/bithumb`, { recursive: true });
  const started = new Date().toISOString();

  const probes = await Promise.all(
    [
      "https://api.bithumb.com/v1/market/all",
      "https://api.binance.com/api/v3/ping",
      "https://data-api.binance.vision/api/v3/ping",
      "https://fapi.binance.com/fapi/v1/ping",
      "https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1d/",
      "https://api.coingecko.com/api/v3/ping",
    ].map(probe),
  );
  console.log("접속 시험:", probes.map((p) => `${p.status} ${p.url}`).join("\n  "));

  const markets = (await getJson("https://api.bithumb.com/v1/market/all?isDetails=true")).filter((m) => m.market.startsWith("KRW-"));
  console.log(`빗썸 원화 마켓 ${markets.length}개`);
  toFormat = await pickToFormat();
  console.log(toFormat ? `과거 페이지 형식: ${toFormat}` : "과거 페이지를 넘길 방법을 찾지 못해 최근 200일만 받는다");

  const queue = [...markets];
  const done = [];
  const failed = [];
  await Promise.all(
    Array.from({ length: WORKERS }, async () => {
      for (let m = queue.shift(); m; m = queue.shift()) {
        try {
          const rows = await bithumbDaily(m.market);
          const sym = m.market.slice(4);
          await writeFile(
            `${OUT}/bithumb/${sym}.json`,
            JSON.stringify({ sym, name: m.korean_name, en: m.english_name, warning: m.market_warning ?? "NONE", rows }),
          );
          done.push({ sym, days: rows.length, first: rows[0]?.[0], last: rows.at(-1)?.[0], missing: missingDays(rows) });
          if (done.length % 50 === 0) console.log(`  ${done.length}/${markets.length}`);
        } catch (e) {
          failed.push({ market: m.market, error: String(e?.message ?? e) });
        }
      }
    }),
  );

  let binance = null;
  try {
    binance = await collectBinance();
  } catch (e) {
    console.log(`바이낸스 수집 실패: ${String(e?.message ?? e).slice(0, 200)}`);
    binance = { coins: 0, error: String(e?.message ?? e) };
  }

  const gaps = done.filter((d) => d.missing > 0);
  const checks = {
    bithumbCandleTimesKst: [...candleTimes],
    bithumbCoinsWithMissingDays: gaps.length,
    bithumbMaxMissing: Math.max(0, ...gaps.map((d) => d.missing)),
    binanceCoinsWithMissingDays: binance?.coinsWithMissingDays ?? null,
  };
  console.log("점검:", JSON.stringify(checks));
  const meta = { started, finished: new Date().toISOString(), probes, toFormat, checks, coins: done.length, failed, binance, list: done.sort((a, b) => a.sym.localeCompare(b.sym)) };
  await writeFile(`${OUT}/meta.json`, JSON.stringify(meta, null, 1));
  const lens = done.map((d) => d.days).sort((a, b) => a - b);
  const btc = done.find((d) => d.sym === "BTC");
  console.log(`완료: ${done.length}개, 실패 ${failed.length}개 · 봉 수 중앙값 ${lens[lens.length >> 1]} · BTC ${btc?.first} ~ ${btc?.last}`);
  if (done.length === 0) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
