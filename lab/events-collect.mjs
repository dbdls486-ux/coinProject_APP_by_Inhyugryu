#!/usr/bin/env node
/**
 * 사건 자료 수집 — 날짜가 객관적으로 찍히는 재료만 모은다 (공식 공개 API 만 사용).
 *  - 업비트 원화 상장: 종목별 첫 거래 시각 (api.upbit.com 캔들)
 *  - 바이낸스 현물 상장: 첫 1시간봉 시각 (data-api.binance.vision)
 *  - 바이낸스 선물 상장: 펀딩비 첫 날짜 (이미 받아 둔 자료실 파일)
 *  - 빗썸 원화 상장: 일봉 첫 날짜 (이미 받아 둔 파일)
 *  - 해킹: DefiLlama 공개 목록 (프로토콜 이름을 코인 영문명과 맞춤)
 * 결과: lab/.data/events.json — [{ type, sym, ts(밀리초, 이 시각에는 확실히 공개된 상태) }]
 */
import { readdir, readFile, writeFile } from "node:fs/promises";

const OUT = process.env.LAB_DATA ?? "lab/.data";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const DAY = 86_400_000;

async function getJson(url, tries = 5) {
  let last;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, { headers: { Accept: "application/json" } });
      if (res.status === 429 || res.status >= 500) {
        last = new Error(`HTTP ${res.status}`);
        await sleep(1500 * (i + 1));
        continue;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      last = e;
      await sleep(800 * (i + 1));
    }
  }
  throw last;
}

async function pool(items, n, gapMs, fn) {
  const q = [...items];
  await Promise.all(
    Array.from({ length: n }, async () => {
      for (let x = q.shift(); x !== undefined; x = q.shift()) {
        try {
          await fn(x);
        } catch {
          /* 한 종목 실패는 건너뜀 */
        }
        await sleep(gapMs);
      }
    }),
  );
}

const events = [];
const names = new Map(); // 심볼 → 영문명 (해킹 이름 맞추기용)
const note = {};

/* ---------- 업비트 ---------- */
try {
  const markets = (await getJson("https://api.upbit.com/v1/market/all?is_details=false")).filter((m) => m.market.startsWith("KRW-"));
  for (const m of markets) names.set(m.market.slice(4), m.english_name);
  let n = 0;
  const iso = (ms) => new Date(ms).toISOString().slice(0, 19) + "Z";
  await pool(markets, 3, 120, async (m) => {
    // 1) 월봉으로 첫 달을 찾고 2) 그 달의 일봉으로 첫 거래일을 찾고 3) 그날의 1시간봉으로 첫 거래 시각을 찾는다.
    //    (월봉의 날짜는 그 달 1일이라 그대로 쓰면 실제 상장보다 앞선 날짜가 된다)
    const months = await getJson(`https://api.upbit.com/v1/candles/months?market=${m.market}&count=200`);
    const firstMonth = months.at(-1)?.candle_date_time_utc;
    if (!firstMonth) return;
    await sleep(120);
    const days = await getJson(`https://api.upbit.com/v1/candles/days?market=${m.market}&to=${encodeURIComponent(iso(Date.parse(firstMonth + "Z") + 33 * DAY))}&count=40`);
    const firstDay = days.at(-1)?.candle_date_time_utc;
    if (!firstDay) return;
    const dayStart = Date.parse(firstDay + "Z");
    await sleep(120);
    const hours = await getJson(`https://api.upbit.com/v1/candles/minutes/60?market=${m.market}&to=${encodeURIComponent(iso(dayStart + DAY))}&count=30`);
    const firstHour = hours.map((h) => Date.parse(h.candle_date_time_utc + "Z")).filter((t) => t >= dayStart && t < dayStart + DAY).sort((a, b) => a - b)[0];
    // 시각을 못 찾으면 그날이 끝난 뒤에 알려진 것으로 본다 (보수적으로)
    events.push({ type: "LIST_UPBIT", sym: m.market.slice(4), ts: firstHour !== undefined ? firstHour + 3_600_000 : dayStart + DAY, exact: firstHour !== undefined });
    n++;
  });
  note.upbit = `${n}/${markets.length}`;
} catch (e) {
  note.upbitError = String(e?.message ?? e);
}

/* ---------- 바이낸스 현물 ---------- */
try {
  const files = (await readdir(`${OUT}/binance`)).filter((f) => f.endsWith(".json"));
  let exact = 0;
  await pool(files, 4, 60, async (f) => {
    const j = JSON.parse(await readFile(`${OUT}/binance/${f}`, "utf8"));
    if (!j.rows?.length) return;
    let ts = Date.parse(j.rows[0][0] + "T00:00:00Z") + DAY; // 날짜만 알 때는 그날 끝
    if (!j.delisted) {
      try {
        const k = await getJson(`https://data-api.binance.vision/api/v3/klines?symbol=${j.sym}USDT&interval=1h&startTime=0&limit=1`, 2);
        if (k?.[0]?.[0]) {
          ts = k[0][0] + 3_600_000;
          exact++;
        }
      } catch {
        /* 날짜 기준 유지 */
      }
    }
    events.push({ type: "LIST_BINANCE", sym: j.sym, ts });
  });
  note.binance = `${files.length}개 (시각까지 ${exact}개)`;
} catch (e) {
  note.binanceError = String(e?.message ?? e);
}

/* ---------- 바이낸스 선물 ---------- */
try {
  const files = (await readdir(`${OUT}/funding`)).filter((f) => f.endsWith(".json"));
  for (const f of files) {
    const j = JSON.parse(await readFile(`${OUT}/funding/${f}`, "utf8"));
    if (j.rows?.length) events.push({ type: "LIST_PERP", sym: j.sym, ts: Date.parse(j.rows[0][0] + "T00:00:00Z") + DAY });
  }
  note.perp = files.length;
} catch (e) {
  note.perpError = String(e?.message ?? e);
}

/* ---------- 빗썸 ---------- */
try {
  const files = (await readdir(`${OUT}/bithumb`)).filter((f) => f.endsWith(".json"));
  for (const f of files) {
    const j = JSON.parse(await readFile(`${OUT}/bithumb/${f}`, "utf8"));
    if (!j.rows?.length) continue;
    if (j.en && !names.has(j.sym)) names.set(j.sym, j.en);
    // 빗썸 일봉 날짜는 한국 날짜 → 그날 끝 = 같은 날짜 15:00 UTC
    events.push({ type: "LIST_BITHUMB", sym: j.sym, ts: Date.parse(j.rows[0][0] + "T15:00:00Z") });
  }
  note.bithumb = files.length;
} catch (e) {
  note.bithumbError = String(e?.message ?? e);
}

/* ---------- 해킹 ---------- */
try {
  const hacks = await getJson("https://api.llama.fi/hacks");
  const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
  const SUFFIX = /(network|protocol|finance|bridge|dao|labs|chain|swap|exchange|token|foundation|wallet)$/;
  const byName = new Map();
  for (const [sym, en] of names) {
    const n = norm(en);
    if (n.length >= 4) byName.set(n, sym);
  }
  let matched = 0;
  for (const h of hacks) {
    if (!(h.amount >= 1e6) || !h.date) continue;
    const full = norm(h.name);
    const sym = byName.get(full) ?? byName.get(full.replace(SUFFIX, ""));
    if (!sym) continue;
    events.push({ type: "HACK", sym, ts: h.date * 1000 + DAY, amount: h.amount, name: h.name });
    matched++;
  }
  note.hacks = `${matched}/${hacks.length}`;
} catch (e) {
  note.hacksError = String(e?.message ?? e);
}

/* ---------- 업비트 상장 전후 시간 단위 가격 (바이낸스 1시간봉) ---------- */
// 공지는 첫 거래보다 몇 시간 앞서 나온다. 가격이 언제 뛰는지 보려고 첫 거래 24시간 전 ~ 72시간 후를 받는다.
const paths = [];
try {
  const listed = new Set((await readdir(`${OUT}/binance`)).map((f) => f.replace(/\.json$/, "")));
  const targets = events.filter((e) => e.type === "LIST_UPBIT" && e.exact && e.ts >= Date.UTC(2019, 6, 1) && listed.has(e.sym) && e.sym !== "BTC");
  const kl = async (symbol, start) => (await getJson(`https://data-api.binance.vision/api/v3/klines?symbol=${symbol}&interval=1h&startTime=${start}&limit=104`, 2)).map((k) => [k[0], +k[1], +k[4]]);
  await pool(targets, 3, 80, async (e) => {
    const t0 = e.ts - 3_600_000; // 첫 거래가 있던 1시간봉의 시작
    const start = t0 - 25 * 3_600_000;
    const coin = await kl(`${e.sym}USDT`, start);
    if (coin.length < 60 || coin[0][0] > start + 3_600_000) return; // 상장 전부터 바이낸스에 있던 코인만
    const btc = await kl("BTCUSDT", start);
    paths.push({ sym: e.sym, t0, coin, btc });
  });
  note.paths = `${paths.length}/${targets.length}`;
} catch (e) {
  note.pathsError = String(e?.message ?? e);
}

/* ---------- 메이저 코인 4시간봉 (바이낸스) ---------- */
try {
  const { mkdir } = await import("node:fs/promises");
  await mkdir(`${OUT}/h4`, { recursive: true });
  let saved = 0;
  for (const sym of ["BTC", "ETH", "XRP", "SOL", "DOGE"]) {
    const rows = [];
    let start = Date.UTC(2019, 0, 1);
    for (let page = 0; page < 40; page++) {
      const list = await getJson(`https://data-api.binance.vision/api/v3/klines?symbol=${sym}USDT&interval=4h&startTime=${start}&limit=1000`);
      await sleep(80);
      if (!Array.isArray(list) || list.length === 0) break;
      for (const k of list) rows.push([k[0], +k[1], +k[2], +k[3], +k[4], +k[7]]);
      if (list.length < 1000) break;
      start = list.at(-1)[0] + 4 * 3_600_000;
    }
    if (rows.length) {
      await writeFile(`${OUT}/h4/${sym}.json`, JSON.stringify({ sym, rows }));
      saved++;
    }
  }
  note.h4 = saved;
} catch (e) {
  note.h4Error = String(e?.message ?? e);
}

await writeFile(`${OUT}/events.json`, JSON.stringify({ collectedAt: new Date().toISOString(), note, events, paths }));
const count = {};
for (const e of events) count[e.type] = (count[e.type] ?? 0) + 1;
// 점검: 날짜가 매달 1~2일에 몰려 있으면 실제 날짜가 아니라 달의 시작으로 잡힌 것이다
for (const type of Object.keys(count)) {
  const list = events.filter((e) => e.type === type);
  const early = list.filter((e) => new Date(e.ts).getUTCDate() <= 2).length;
  note[`${type}_1~2일비율`] = Math.round((early / list.length) * 100) + "%";
}
for (const sym of ["LPT", "PEPE", "WLD", "ENA"]) {
  const u = events.find((e) => e.type === "LIST_UPBIT" && e.sym === sym);
  if (u) note[`upbit_${sym}`] = new Date(u.ts).toISOString().slice(0, 16);
}
console.log("사건 수집:", JSON.stringify(count), JSON.stringify(note));
