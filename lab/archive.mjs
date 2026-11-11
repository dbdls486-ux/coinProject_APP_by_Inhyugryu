/**
 * 바이낸스 공개 자료실(data.binance.vision) 수집 — 일반 API 가 막힌 미국 서버에서도 열린다.
 *  - 상장폐지된 USDT 현물 종목의 과거 일봉 (생존 편향 줄이기)
 *  - USDT 무기한 선물의 펀딩비 (하루 평균)
 * 자료실은 월별 zip 파일이다. 목록은 S3 목록 API, 압축은 러너의 unzip 으로 푼다.
 */
import { execFileSync } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";

const S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision";
const FILES = "https://data.binance.vision";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchRetry(url, tries = 5) {
  let last;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url);
      if (res.status === 404) return null;
      if (res.ok) return res;
      last = new Error(`HTTP ${res.status}`);
    } catch (e) {
      last = e;
    }
    await sleep(500 * (i + 1));
  }
  throw last;
}

/** S3 목록: prefix 아래 하위 폴더(CommonPrefixes)와 파일(Key) */
async function list(prefix) {
  const dirs = [];
  const keys = [];
  let marker = "";
  for (let page = 0; page < 200; page++) {
    const q = new URLSearchParams({ delimiter: "/", prefix });
    if (marker) q.set("marker", marker);
    const res = await fetchRetry(`${S3}?${q}`);
    if (!res) break;
    const xml = await res.text();
    const d = [...xml.matchAll(/<Prefix>([^<]+)<\/Prefix>/g)].map((m) => m[1]).filter((p) => p !== prefix);
    const k = [...xml.matchAll(/<Key>([^<]+)<\/Key>/g)].map((m) => m[1]);
    dirs.push(...d);
    keys.push(...k);
    if (!/<IsTruncated>true<\/IsTruncated>/.test(xml)) break;
    marker = xml.match(/<NextMarker>([^<]+)<\/NextMarker>/)?.[1] ?? [...d, ...k].at(-1);
    if (!marker) break;
  }
  return { dirs, keys };
}

let tmpSeq = 0;
/** zip 하나를 받아 첫 파일(CSV) 내용을 돌려준다 */
async function csvFromZip(key) {
  const res = await fetchRetry(`${FILES}/${key}`);
  if (!res) return "";
  const buf = Buffer.from(await res.arrayBuffer());
  const tmp = `${tmpdir()}/lab-${process.pid}-${tmpSeq++}.zip`;
  await writeFile(tmp, buf);
  try {
    return execFileSync("unzip", ["-p", tmp], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  } finally {
    try {
      execFileSync("rm", ["-f", tmp]);
    } catch {
      /* 무시 */
    }
  }
}

/** 동시 n개로 작업 큐 처리 */
async function pool(items, n, fn) {
  const q = [...items];
  await Promise.all(
    Array.from({ length: n }, async () => {
      for (let x = q.shift(); x !== undefined; x = q.shift()) await fn(x);
    }),
  );
}

/** 2025년부터 현물 자료의 시각이 마이크로초로 바뀌었다 */
const toMs = (v) => {
  const x = Number(v);
  return x > 1e14 ? Math.floor(x / 1000) : x;
};

/**
 * 상장폐지(지금 거래 중이 아닌) USDT 현물 종목의 일봉.
 * @param {Set<string>} currentBases 지금 거래 중인 기초 자산 (같은 이름이면 건너뜀)
 */
export async function collectDelisted(out, currentBases, stableRe) {
  await mkdir(`${out}/binance`, { recursive: true });
  const { dirs } = await list("data/spot/monthly/klines/");
  const syms = dirs.map((d) => d.split("/").at(-2)).filter((s) => s.endsWith("USDT"));
  const targets = syms
    .map((s) => ({ symbol: s, base: s.slice(0, -4) }))
    .filter(({ base }) => base && !currentBases.has(base) && !stableRe.test(base) && !/(UP|DOWN|BULL|BEAR)$/.test(base));
  console.log(`자료실 USDT 현물 ${syms.length}개 중 상장폐지 ${targets.length}개`);
  const done = [];
  await pool(targets, 8, async ({ symbol, base }) => {
    try {
      const { keys } = await list(`data/spot/monthly/klines/${symbol}/1d/`);
      const zips = keys.filter((k) => k.endsWith(".zip"));
      const rows = new Map();
      for (const key of zips) {
        const csv = await csvFromZip(key);
        for (const line of csv.split("\n")) {
          const c = line.split(",");
          if (c.length < 8 || !/^\d/.test(c[0])) continue;
          const d = new Date(toMs(c[0])).toISOString().slice(0, 10);
          rows.set(d, [d, +c[1], +c[2], +c[3], +c[4], +c[7]]);
        }
      }
      const sorted = [...rows.values()].sort((a, b) => (a[0] < b[0] ? -1 : 1));
      if (sorted.length < 30) return;
      await writeFile(`${out}/binance/${base}.json`, JSON.stringify({ sym: base, name: base, delisted: true, rows: sorted }));
      done.push({ sym: base, days: sorted.length, first: sorted[0][0], last: sorted.at(-1)[0] });
    } catch (e) {
      /* 한 종목 실패는 건너뛴다 */
    }
  });
  console.log(`상장폐지 종목 ${done.length}개 저장`);
  return { delisted: done.length };
}

/**
 * 선물 펀딩비 → 날짜(UTC)별 평균. 1000PEPEUSDT 같은 묶음 단위 선물은 PEPE 로 맞춘다.
 * @param {Set<string>} bases 필요한 기초 자산
 */
export async function collectFunding(out, bases) {
  await mkdir(`${out}/funding`, { recursive: true });
  const { dirs } = await list("data/futures/um/monthly/fundingRate/");
  const syms = dirs.map((d) => d.split("/").at(-2)).filter((s) => s.endsWith("USDT"));
  const norm = (s) => s.slice(0, -4).replace(/^(1000000|1000|1M)(?=[A-Z])/, "");
  const targets = syms.filter((s) => bases.has(norm(s)));
  console.log(`자료실 선물 펀딩비 ${syms.length}개 중 대상 ${targets.length}개`);
  let saved = 0;
  await pool(targets, 8, async (symbol) => {
    try {
      const { keys } = await list(`data/futures/um/monthly/fundingRate/${symbol}/`);
      const sum = new Map();
      for (const key of keys.filter((k) => k.endsWith(".zip"))) {
        const csv = await csvFromZip(key);
        for (const line of csv.split("\n")) {
          const c = line.split(",");
          if (c.length < 3 || !/^\d/.test(c[0])) continue;
          const d = new Date(toMs(c[0])).toISOString().slice(0, 10);
          const r = Number(c[2]);
          if (!Number.isFinite(r)) continue;
          const s = sum.get(d) ?? [0, 0];
          s[0] += r;
          s[1]++;
          sum.set(d, s);
        }
      }
      const rows = [...sum.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1)).map(([d, [s, n]]) => [d, s / n]);
      if (rows.length === 0) return;
      await writeFile(`${out}/funding/${norm(symbol)}.json`, JSON.stringify({ sym: norm(symbol), symbol, rows }));
      saved++;
    } catch {
      /* 건너뜀 */
    }
  });
  console.log(`펀딩비 ${saved}개 저장`);
  return { funding: saved };
}
