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

/** 한 종목의 일봉을 최신 → 과거로 페이지를 넘기며 받는다. 반환은 오래된 순 */
async function bithumbDaily(market) {
  const rows = new Map();
  let to = "";
  for (let page = 0; page < MAX_PAGES; page++) {
    const q = new URLSearchParams({ market, count: "200" });
    if (to) q.set("to", to);
    const list = await getJson(`https://api.bithumb.com/v1/candles/days?${q}`);
    await sleep(PAUSE_MS);
    if (!Array.isArray(list) || list.length === 0) break;
    for (const k of list) {
      const d = String(k.candle_date_time_kst ?? k.candle_date_time_utc).slice(0, 10);
      // [날짜, 시가, 고가, 저가, 종가, 거래대금(원)]
      rows.set(d, [d, k.opening_price, k.high_price, k.low_price, k.trade_price, k.candle_acc_trade_price ?? 0]);
    }
    const oldest = list[list.length - 1].candle_date_time_utc;
    const next = `${oldest}Z`;
    if (!oldest || next === to) break;
    to = next;
    if (list.length < 200) break;
  }
  return [...rows.values()].sort((a, b) => (a[0] < b[0] ? -1 : 1));
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
          done.push({ sym, days: rows.length, first: rows[0]?.[0], last: rows.at(-1)?.[0] });
          if (done.length % 50 === 0) console.log(`  ${done.length}/${markets.length}`);
        } catch (e) {
          failed.push({ market: m.market, error: String(e?.message ?? e) });
        }
      }
    }),
  );

  const meta = { started, finished: new Date().toISOString(), probes, coins: done.length, failed, list: done.sort((a, b) => a.sym.localeCompare(b.sym)) };
  await writeFile(`${OUT}/meta.json`, JSON.stringify(meta, null, 1));
  console.log(`완료: ${done.length}개, 실패 ${failed.length}개`);
  if (done.length === 0) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
