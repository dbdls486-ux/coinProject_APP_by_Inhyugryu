#!/usr/bin/env node
/**
 * 사건·국면·메이저 타이밍 실험 (4차)
 *  1. 상장·해킹 같은 사건이 공개된 뒤 다음 날 시가에 샀을 때 (같은 날·같은 변동성 코인 평균 대비)
 *  2. 사건 전후 하루 단위 반응 (당일 반응은 이미 지나간 것이라 살 수 없다)
 *  3. 그 시장에 처음 상장된 코인을 다음 날 사서 들고 있었을 때
 *  4. 급등한 다음 날 샀을 때 + 급등 사례 목록 (재료 조사용)
 *  5. 차트·수급 규칙을 BTC 50일선 위/아래 국면으로 나눠 본 선택력
 *  6. 메이저 코인 타이밍 규칙 (추세 vs 싸면 사고 비싸면 팔기)
 * 결과: lab/results/EVENTS.md, lab/results/events.json
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { DATA, MARKETS, OUT, SPLIT, periods, prepareAll } from "./common.mjs";
import { selectionStats, tradeTable } from "./selection.mjs";
import { EXITS, STRATEGIES } from "./strategies.mjs";
import { MAJORS, TIMING_RULES, h4Timing, majorsTiming } from "./timing.mjs";

const DAY = 86_400_000;
const EVENT_TYPES = [
  { id: "LIST_UPBIT", label: "업비트 원화 상장", minTs: Date.UTC(2018, 0, 1) },
  { id: "LIST_BITHUMB", label: "빗썸 원화 상장", minTs: Date.UTC(2019, 0, 1) },
  { id: "LIST_BINANCE", label: "바이낸스 현물 상장", minTs: Date.UTC(2018, 0, 1) },
  { id: "LIST_PERP", label: "바이낸스 선물 상장", minTs: Date.UTC(2020, 0, 1) },
  { id: "HACK", label: "해킹 피해 (100만 달러 이상)", minTs: 0 },
];

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
const median = (xs) => {
  if (!xs.length) return NaN;
  const a = [...xs].sort((x, y) => x - y);
  return a.length % 2 ? a[(a.length - 1) / 2] : (a[a.length / 2 - 1] + a[a.length / 2]) / 2;
};
const tPlain = (xs) => {
  if (xs.length < 5) return NaN;
  const m = mean(xs);
  const sd = Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1));
  return sd > 0 ? m / (sd / Math.sqrt(xs.length)) : NaN;
};

/** 사건 → 그 시장에서 "이 봉 종가에는 확실히 공개돼 있던" 봉에 표시 */
function mapEvents(U, M, events) {
  const bySym = new Map(U.coins.map((k) => [k.sym, k]));
  const barEnd = (k, i) => Date.parse(U.dates[k.day[i]] + "T00:00:00Z") + M.dayEndUtcH * 3_600_000;
  const flags = new Map(EVENT_TYPES.map((e) => [e.id, new Map()]));
  const minTs = new Map(EVENT_TYPES.map((e) => [e.id, e.minTs]));
  for (const ev of events) {
    if (!flags.has(ev.type) || ev.ts < minTs.get(ev.type)) continue;
    const k = bySym.get(ev.sym);
    if (!k) continue;
    // 이분 탐색: 봉이 끝나는 시각이 사건 시각 이후인 첫 봉
    let lo = 0;
    let hi = k.n;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (barEnd(k, mid) >= ev.ts) hi = mid;
      else lo = mid + 1;
    }
    if (lo >= k.n || lo < 1 || barEnd(k, lo) - ev.ts > 2 * DAY) continue;
    const m = flags.get(ev.type);
    if (!m.has(k)) m.set(k, new Uint8Array(k.n));
    m.get(k)[lo] = 1;
  }
  return flags;
}

/** 날짜별 거래 가능 종목의 하루 수익률 평균·중앙값 */
function dailyCross(U) {
  const meanR = new Float64Array(U.T).fill(NaN);
  const medR = new Float64Array(U.T).fill(NaN);
  for (let t = 0; t < U.T; t++) {
    const rs = [];
    for (const [k, i] of U.live[t]) if (i >= 1) rs.push(k.c[i] / k.c[i - 1] - 1);
    if (rs.length >= 5) {
      meanR[t] = mean(rs);
      medR[t] = median(rs);
    }
  }
  return { meanR, medR };
}

/** 사건 전후 하루 단위 초과수익 (시장 평균 대비) */
function reaction(U, flagMap, cross, from, to) {
  const offs = [-3, -2, -1, 0, 1, 2, 3];
  const per = new Map(offs.map((o) => [o, []]));
  const cum7 = [];
  let n = 0;
  for (const [k, f] of flagMap) {
    for (let i = 0; i < k.n; i++) {
      if (!f[i] || !k.eligible[i] || k.day[i] < from || k.day[i] > to) continue;
      n++;
      for (const o of offs) {
        const j = i + o;
        if (j < 1 || j >= k.n) continue;
        const m = cross.meanR[k.day[j]];
        if (Number.isFinite(m)) per.get(o).push(k.c[j] / k.c[j - 1] - 1 - m);
      }
      if (i + 7 < k.n) {
        let s = 0;
        for (let j = i + 1; j <= i + 7; j++) {
          const m = cross.meanR[k.day[j]];
          s += k.c[j] / k.c[j - 1] - 1 - (Number.isFinite(m) ? m : 0);
        }
        cum7.push(s);
      }
    }
  }
  return { n, days: offs.map((o) => ({ offset: o, mean: mean(per.get(o)), median: median(per.get(o)) })), cum7: { mean: mean(cum7), median: median(cum7), t: tPlain(cum7) } };
}

/** 그 시장에 처음 상장된 코인: 둘째 날 시가에 사서 h일 뒤 종가까지, 같은 기간 BTC 대비 */
function newListingDrift(U, P) {
  const btc = U.btc;
  const btcAt = (t, field) => {
    for (let s = t; s >= 0; s--) if (btc.idx[s] >= 0) return btc[field][btc.idx[s]];
    return NaN;
  };
  const out = [];
  for (const [label, a, b] of [["고르는 기간", P.start, P.split - 1], ["확인 기간", P.split, P.end]]) {
    for (const h of [7, 14, 28]) {
      const ex = [];
      const raw = [];
      const day1 = [];
      for (const k of U.coins) {
        if (k === btc || k.day[0] < a || k.day[0] > b || k.n < h + 3) continue;
        const r = k.c[1 + h] / k.o[1] - 1;
        const rb = btcAt(k.day[1 + h], "c") / btcAt(k.day[1], "o") - 1;
        if (!Number.isFinite(r) || !Number.isFinite(rb)) continue;
        raw.push(r);
        ex.push(r - rb);
        day1.push(k.c[0] / k.o[0] - 1);
      }
      out.push({ period: label, hold: h, n: ex.length, meanRaw: mean(raw), medianRaw: median(raw), meanExcess: mean(ex), medianExcess: median(ex), positive: ex.filter((x) => x > 0).length / Math.max(1, ex.length), t: tPlain(ex), firstDayMedian: median(day1) });
    }
  }
  return out;
}

/** 급등 사례 목록 (확인 기간, 시장 중앙값 대비 초과 상승이 큰 순) */
function pumpList(U, P, cross, minExcess, limit) {
  const btc = U.btc;
  const btcAt = (t, field) => {
    for (let s = t; s >= 0; s--) if (btc.idx[s] >= 0) return btc[field][btc.idx[s]];
    return NaN;
  };
  const rows = [];
  for (let t = P.split; t <= P.end; t++) {
    for (const [k, i] of U.live[t]) {
      if (i < 1 || k === btc) continue;
      const r = k.c[i] / k.c[i - 1] - 1;
      const ex = r - cross.medR[t];
      if (!(ex >= minExcess)) continue;
      const fwd = (h) => (i + h < k.n ? k.c[i + h] / k.o[i + 1] - 1 : null);
      const fb = (h) => (i + h < k.n ? btcAt(k.day[i + h], "c") / btcAt(k.day[i + 1], "o") - 1 : null);
      rows.push({ date: U.dates[t], sym: k.sym, name: k.name, dayRet: r, excess: ex, volRatio: k.f.volRatio[i], val20: k.f.val20[i], btcUp: U.btcUp[t], fwd1: fwd(1), fwd3: fwd(3), fwd7: fwd(7), fwd14: fwd(14), btc7: fb(7), btc14: fb(14) });
    }
  }
  rows.sort((a, b) => b.excess - a.excess);
  // 같은 코인이 5일 안에 연달아 잡히면 가장 큰 것만
  const kept = [];
  for (const r of rows) {
    if (kept.some((x) => x.sym === r.sym && Math.abs(Date.parse(x.date) - Date.parse(r.date)) <= 5 * DAY)) continue;
    kept.push(r);
    if (kept.length >= limit) break;
  }
  return kept;
}

async function main() {
  const t0 = Date.now();
  const { events, note, paths = [] } = JSON.parse(await readFile(`${DATA}/events.json`, "utf8"));
  const { U: Us } = await prepareAll(undefined);
  const result = { generatedAt: new Date().toISOString(), collectNote: note, markets: [] };

  for (const M of MARKETS) {
    const U = Us[M.id];
    if (!U) continue;
    const P = periods(U);
    const cross = dailyCross(U);
    const flags = mapEvents(U, M, events);
    const up = U.btcUp;
    const down = Uint8Array.from(up, (v) => (v ? 0 : 1));
    const tables = Object.fromEntries(EXITS.map((e) => [e.id, tradeTable(U, e, M.sim)]));

    // 1. 사건 뒤 매수
    const eventRows = [];
    for (const E of EVENT_TYPES) {
      const fm = flags.get(E.id);
      const strat = { id: E.id, entry: (k, i) => (fm.get(k)?.[i] ? 1 : null) };
      for (const exit of EXITS) {
        const T = tables[exit.id];
        eventRows.push({
          type: E.id,
          label: E.label,
          exit: exit.id,
          exitLabel: exit.label,
          train: selectionStats(U, strat, T, P.start, P.split - 2 - exit.maxHold, null, 10, true),
          test: selectionStats(U, strat, T, P.split, P.end, null, 10, true),
          all: selectionStats(U, strat, T, P.start, P.end, null, 10, true),
          btcUp: selectionStats(U, strat, T, P.start, P.end, up, 10, true),
          btcDown: selectionStats(U, strat, T, P.start, P.end, down, 10, true),
        });
      }
    }
    // 사건별 내역 (어떤 상장이 올랐고 어떤 상장이 빠졌는지 보려고)
    const eventList = [];
    for (const E of EVENT_TYPES) {
      for (const [k, f] of flags.get(E.id)) {
        for (let i = 1; i < k.n; i++) {
          if (!f[i] || !k.eligible[i] || k.day[i] < P.start) continue;
          const t = k.day[i];
          const b = k.bucket[i];
          const ex = (id) => {
            const r = tables[id].table.get(k)[i];
            const mu = b >= 0 ? tables[id].mean[t * 5 + b] : NaN;
            return Number.isFinite(r) && Number.isFinite(mu) ? r - mu : null;
          };
          const raw = tables.H14.table.get(k)[i];
          eventList.push({ type: E.id, sym: k.sym, name: k.name, date: U.dates[t], btcUp: up[t], day0: k.c[i] / k.c[i - 1] - 1 - cross.medR[t], pre3: i >= 4 ? k.c[i - 1] / k.c[i - 4] - 1 : null, ex7: ex("H7"), ex14: ex("H14"), ex28: ex("H28"), raw14: Number.isFinite(raw) ? raw : null });
        }
      }
    }
    eventList.sort((a, b) => (a.date < b.date ? -1 : 1));
    // 2. 사건 전후 반응
    const reactions = EVENT_TYPES.map((E) => ({ type: E.id, label: E.label, ...reaction(U, flags.get(E.id), cross, P.start, P.end) }));
    // 3. 신규 상장 코인
    const listing = newListingDrift(U, P);
    // 4. 급등 다음 날
    const pumpRows = [];
    for (const thr of [0.1, 0.2, 0.4]) {
      const strat = { id: `PUMP${thr}`, entry: (k, i, t) => (i >= 1 && k.c[i] / k.c[i - 1] - 1 - cross.medR[t] >= thr ? k.c[i] / k.c[i - 1] : null) };
      for (const exit of EXITS) {
        const T = tables[exit.id];
        pumpRows.push({
          threshold: thr,
          exit: exit.id,
          exitLabel: exit.label,
          train: selectionStats(U, strat, T, P.start, P.split - 2 - exit.maxHold),
          test: selectionStats(U, strat, T, P.split, P.end),
          btcUp: selectionStats(U, strat, T, P.start, P.end, up),
          btcDown: selectionStats(U, strat, T, P.start, P.end, down),
        });
      }
    }
    const pumps = pumpList(U, P, cross, 0.25, 150);
    for (const p of pumps) {
      const end = Date.parse(p.date + "T00:00:00Z") + M.dayEndUtcH * 3_600_000;
      p.events = [...new Set(events.filter((e) => e.sym === p.sym && e.type !== "LIST_" + M.id.toUpperCase() && Math.abs(e.ts - end) <= 2 * DAY).map((e) => e.type))];
    }
    // 4-2. 급등의 성격별 시나리오: 시장 전체가 같이 오른 날인지, 혼자 오른 것인지, 거래량 쏠림, 급등 뒤 눌림
    const pumpEx = (k, i, t) => (i >= 1 ? k.c[i] / k.c[i - 1] - 1 - cross.medR[t] : NaN);
    const isPump = (k, i, t) => pumpEx(k, i, t) >= 0.2;
    const SCEN = [
      { id: "PUMP_BROAD", label: "급등 + 그날 시장 중앙값도 +3% 이상 (시장 동반 상승)", entry: (k, i, t) => (isPump(k, i, t) && cross.medR[t] >= 0.03 ? 1 : null) },
      { id: "PUMP_ALONE", label: "급등 + 그날 시장 중앙값 +1% 미만 (혼자 급등)", entry: (k, i, t) => (isPump(k, i, t) && cross.medR[t] < 0.01 ? 1 : null) },
      { id: "PUMP_VOL20", label: "급등 + 거래대금 평소 20배 이상", entry: (k, i, t) => (isPump(k, i, t) && k.f.volRatio[i] >= 20 ? 1 : null) },
      { id: "PUMP_VOL5", label: "급등 + 거래대금 평소 5배 미만", entry: (k, i, t) => (isPump(k, i, t) && k.f.volRatio[i] < 5 ? 1 : null) },
      { id: "PUMP_TREND", label: "급등 + 급등 전 이미 50일선 위", entry: (k, i, t) => (isPump(k, i, t) && k.c[i - 1] > k.f.sma50[i - 1] ? 1 : null) },
      { id: "PUMP_BASE", label: "급등 + 급등 전 50일선 아래 (바닥권)", entry: (k, i, t) => (isPump(k, i, t) && k.c[i - 1] < k.f.sma50[i - 1] ? 1 : null) },
      {
        id: "PUMP_PULLBACK",
        label: "급등 3~10일 뒤 눌림: 고점 대비 20% 넘게 식었지만 급등일 시가보다 5% 이상 위, 거래대금은 급등일의 절반 미만",
        entry: (k, i, t) => {
          for (let j = i - 3; j >= Math.max(1, i - 10); j--) {
            if (!isPump(k, j, k.day[j])) continue;
            let hi = 0;
            for (let m = j; m <= i; m++) if (k.h[m] > hi) hi = k.h[m];
            return k.c[i] <= hi * 0.8 && k.c[i] >= k.o[j] * 1.05 && k.v[i] < k.v[j] * 0.5 ? 1 : null;
          }
          return null;
        },
      },
    ];
    const scenRows = [];
    for (const st of SCEN) {
      for (const exit of EXITS) {
        const T = tables[exit.id];
        scenRows.push({ id: st.id, label: st.label, exit: exit.id, exitLabel: exit.label, train: selectionStats(U, st, T, P.start, P.split - 2 - exit.maxHold), test: selectionStats(U, st, T, P.split, P.end) });
      }
    }
    // 5. 규칙의 국면별 선택력 (14일 보유)
    const T14 = tables.H14;
    const regimeRows = STRATEGIES.map((st) => ({
      strategy: st.id,
      label: st.label,
      btcUp: selectionStats(U, st, T14, P.start, P.end, up),
      btcDown: selectionStats(U, st, T14, P.start, P.end, down),
    }));
    // 6. 메이저 타이밍
    const majors = majorsTiming(U, P, M.sim.fee);

    result.markets.push({ market: M.id, label: M.label, periods: { train: [U.dates[P.start], U.dates[P.split - 1]], test: [U.dates[P.split], U.dates[P.end]] }, eventRows, eventList, reactions, listing, pumpRows, scenRows, pumps, regimeRows, majors });
    console.log(`[${M.label}] 사건 실험 완료`);
  }

  // 7. 업비트 상장 전후 시간 단위 경로 (바이낸스 1시간봉, BTC 대비)
  {
    const HS = [-24, -12, -6, -4, -3, -2, -1, 0, 1, 2, 4, 8, 12, 24, 48, 72];
    const at = (rows, ts) => rows.find((r) => r[0] === ts);
    const cum = new Map(HS.map((h) => [h, []]));
    const after = { h0: { 24: [], 72: [] }, h1: { 24: [], 72: [] }, h4: { 24: [], 72: [] } };
    let used = 0;
    for (const p of paths) {
      const base = at(p.coin, p.t0 - 25 * 3_600_000) ?? p.coin[0];
      const bbase = at(p.btc, base[0]);
      if (!bbase) continue;
      const rel = (h) => {
        const c = at(p.coin, p.t0 + h * 3_600_000);
        const b = at(p.btc, p.t0 + h * 3_600_000);
        return c && b ? c[2] / base[2] - 1 - (b[2] / bbase[2] - 1) : null; // h 시간째 봉의 종가까지
      };
      used++;
      for (const h of HS) {
        const v = rel(h);
        if (v !== null) cum.get(h).push(v);
      }
      // 첫 거래 시간봉이 끝난 뒤(h0), 1시간 뒤, 4시간 뒤에 샀을 때 24·72시간 뒤까지 (BTC 대비)
      for (const [key, h] of [["h0", 0], ["h1", 1], ["h4", 4]]) {
        const c0 = at(p.coin, p.t0 + h * 3_600_000);
        const b0 = at(p.btc, p.t0 + h * 3_600_000);
        for (const fw of [24, 72]) {
          const c1 = at(p.coin, p.t0 + (h + fw) * 3_600_000);
          const b1 = at(p.btc, p.t0 + (h + fw) * 3_600_000);
          if (c0 && b0 && c1 && b1) after[key][fw].push(c1[2] / c0[2] - 1 - (b1[2] / b0[2] - 1));
        }
      }
    }
    // 빠르게 반응했다면: 첫 거래 전 8시간 안에 1시간 동안 BTC보다 5%p 넘게 뛴 첫 시간봉(공지로 추정)이 끝날 때 사서,
    // 업비트 첫 거래 시간봉이 끝날 때 / 24시간 뒤에 팔았을 때
    const fast = { jump: [], toOpen: [], to24: [], lead: [] };
    for (const p of paths) {
      const ex = (h) => {
        const c1 = at(p.coin, p.t0 + h * 3_600_000);
        const c0 = at(p.coin, p.t0 + (h - 1) * 3_600_000);
        const b1 = at(p.btc, p.t0 + h * 3_600_000);
        const b0 = at(p.btc, p.t0 + (h - 1) * 3_600_000);
        return c1 && c0 && b1 && b0 ? c1[2] / c0[2] - 1 - (b1[2] / b0[2] - 1) : null;
      };
      let hj = null;
      for (let h = -8; h <= -1; h++) {
        const v = ex(h);
        if (v !== null && v >= 0.05) {
          hj = h;
          fast.jump.push(v);
          break;
        }
      }
      if (hj === null) continue;
      const rel = (h1) => {
        const c0 = at(p.coin, p.t0 + hj * 3_600_000);
        const b0 = at(p.btc, p.t0 + hj * 3_600_000);
        const c1 = at(p.coin, p.t0 + h1 * 3_600_000);
        const b1 = at(p.btc, p.t0 + h1 * 3_600_000);
        return c0 && b0 && c1 && b1 ? c1[2] / c0[2] - 1 - (b1[2] / b0[2] - 1) : null;
      };
      const a = rel(0);
      const b = rel(hj + 24);
      if (a !== null) fast.toOpen.push(a);
      if (b !== null) fast.to24.push(b);
      fast.lead.push(-hj);
    }
    const st = (xs) => ({ n: xs.length, mean: mean(xs), median: median(xs), up: xs.filter((x) => x > 0).length / Math.max(1, xs.length) });
    result.upbitFast = { total: used, found: fast.jump.length, medianLeadHours: median(fast.lead), jumpHour: st(fast.jump), toOpen: st(fast.toOpen), to24h: st(fast.to24) };
    result.upbitPath = {
      events: used,
      cumulative: HS.map((h) => ({ hour: h, mean: mean(cum.get(h)), median: median(cum.get(h)), n: cum.get(h).length })),
      buyAfter: Object.entries(after).map(([key, v]) => ({ when: key, h24: { mean: mean(v[24]), median: median(v[24]), up: v[24].filter((x) => x > 0).length / Math.max(1, v[24].length), n: v[24].length }, h72: { mean: mean(v[72]), median: median(v[72]), up: v[72].filter((x) => x > 0).length / Math.max(1, v[72].length), n: v[72].length } })),
    };
  }
  // 8. 메이저 4시간봉 타이밍 규칙 (바이낸스)
  result.h4 = [];
  for (const sym of MAJORS) {
    try {
      const j = JSON.parse(await readFile(`${DATA}/h4/${sym}.json`, "utf8"));
      result.h4.push(...h4Timing(sym, j.rows, Date.parse(SPLIT + "T00:00:00Z"), 0.001));
    } catch {
      /* 자료 없음 */
    }
  }

  await mkdir(OUT, { recursive: true });
  await writeFile(`${OUT}/events.json`, JSON.stringify(result, (_k, v) => (typeof v === "number" && !Number.isInteger(v) ? Math.round(v * 10000) / 10000 : v), 1));
  await writeFile(`${OUT}/EVENTS.md`, report(result));
  console.log(`끝 (${Math.round((Date.now() - t0) / 1000)}초)`);
}

const pc = (x) => (Number.isFinite(x) ? `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}%` : "-");
const pp = (x) => (Number.isFinite(x) ? `${x >= 0 ? "+" : ""}${(x * 100).toFixed(2)}%p` : "-");
const t1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : "-");
const cell = (s) => (s.picks ? `${pp(s.excess)} (t ${t1(s.t)}, ${s.picks}건, 중앙값 ${pp(s.medExcess)}, 실제 ${pc(s.avgRet)})` : "사례 없음");

function report(R) {
  const L = [];
  L.push("# 사건·국면·메이저 타이밍 실험");
  L.push("");
  L.push(`생성 ${R.generatedAt.slice(0, 16).replace("T", " ")} UTC · 사건 수집: ${JSON.stringify(R.collectNote)}`);
  L.push("");
  L.push("표의 값은 \"같은 날 같은 변동성 구간 코인 평균 대비 거래당 초과수익 (t값, 사례 수, 실제 평균 수익)\"이다. 사건이 공개된 뒤 **다음 날 시가**에 사서 2×ATR 손절·기간 만료로 판 결과이고, 수수료와 슬리피지를 뺐다.");
  L.push("");
  for (const m of R.markets) {
    L.push(`## ${m.label}`);
    L.push(`고르는 기간 ${m.periods.train.join(" ~ ")} · 확인 기간 ${m.periods.test.join(" ~ ")}`);
    L.push("");
    L.push("### 1. 사건이 공개된 뒤 매수");
    L.push("| 사건 | 매도 | 고르는 기간 | 확인 기간 | BTC 50일선 위 | BTC 50일선 아래 |");
    L.push("|---|---|---|---|---|---|");
    for (const r of m.eventRows) L.push(`| ${r.label} | ${r.exitLabel} | ${cell(r.train)} | ${cell(r.test)} | ${cell(r.btcUp)} | ${cell(r.btcDown)} |`);
    L.push("");
    L.push("### 2. 사건 전후 하루 단위 반응 (시장 평균 대비, 평균 / 중앙값)");
    L.push("| 사건 | 사례 | 3일 전 | 2일 전 | 1일 전 | 공개된 날 | 1일 뒤 | 2일 뒤 | 3일 뒤 | 이후 7일 누적 (t) |");
    L.push("|---|---|---|---|---|---|---|---|---|---|");
    for (const r of m.reactions) L.push(`| ${r.label} | ${r.n} | ${r.days.map((d) => `${pc(d.mean)} / ${pc(d.median)}`).join(" | ")} | ${pc(r.cum7.mean)} / ${pc(r.cum7.median)} (${t1(r.cum7.t)}) |`);
    L.push("");
    L.push("### 3. 이 시장에 처음 상장된 코인을 둘째 날 시가에 샀다면");
    L.push("| 기간 | 보유 | 사례 | 평균 수익 | 중앙값 | BTC 대비 평균 | BTC 대비 중앙값 | BTC보다 나은 비율 | t | 상장 첫날 등락 중앙값 |");
    L.push("|---|---|---|---|---|---|---|---|---|---|");
    for (const r of m.listing) L.push(`| ${r.period} | ${r.hold}일 | ${r.n} | ${pc(r.meanRaw)} | ${pc(r.medianRaw)} | ${pc(r.meanExcess)} | ${pc(r.medianExcess)} | ${Math.round(r.positive * 100)}% | ${t1(r.t)} | ${pc(r.firstDayMedian)} |`);
    L.push("");
    L.push("### 4. 급등한 다음 날 매수 (그날 시장 중앙값보다 N%p 넘게 오른 코인)");
    L.push("| 급등 기준 | 매도 | 고르는 기간 | 확인 기간 | BTC 50일선 위 | BTC 50일선 아래 |");
    L.push("|---|---|---|---|---|---|");
    for (const r of m.pumpRows) L.push(`| +${Math.round(r.threshold * 100)}%p 이상 | ${r.exitLabel} | ${cell(r.train)} | ${cell(r.test)} | ${cell(r.btcUp)} | ${cell(r.btcDown)} |`);
    L.push("");
    L.push("### 4-2. 급등의 성격별로 나누면 (급등 = 시장 중앙값보다 +20%p 이상)");
    L.push("| 조건 | 매도 | 고르는 기간 | 확인 기간 |");
    L.push("|---|---|---|---|");
    for (const r of m.scenRows) L.push(`| ${r.label} | ${r.exitLabel} | ${cell(r.train)} | ${cell(r.test)} |`);
    L.push("");
    L.push("### 5. 규칙별 선택력을 BTC 국면으로 나누면 (14일 보유, 전체 기간)");
    L.push("| 규칙 | BTC 50일선 위 | BTC 50일선 아래 |");
    L.push("|---|---|---|");
    for (const r of m.regimeRows) L.push(`| ${r.label} | ${cell(r.btcUp)} | ${cell(r.btcDown)} |`);
    L.push("");
    L.push("### 6. 메이저 코인 타이밍 규칙 (연수익 / 최대 낙폭 · 거래 수 · 승률)");
    for (const period of ["train", "test"]) {
      L.push(`**${period === "train" ? "고르는 기간" : "확인 기간"}**`);
      L.push("");
      const syms = MAJORS.filter((s) => m.majors.some((x) => x.sym === s && x[period]));
      L.push(`| 규칙 | 손절 | ${syms.join(" | ")} |`);
      L.push(`|---|---|${syms.map(() => "---").join("|")}|`);
      for (const rule of TIMING_RULES) {
        for (const stop of [0, 0.05]) {
          const rows = syms.map((s) => m.majors.find((x) => x.sym === s && x.rule === rule.id && x.stop === stop));
          if (rows.every((r) => !r)) continue;
          L.push(`| ${rule.label} | ${stop ? "5%" : "없음"} | ${rows.map((r) => (r?.[period] ? `${pc(r[period].cagr)} / ${pc(r[period].mdd)} · ${r[period].trades}회 · ${Math.round(r[period].winRate * 100)}%` : "-")).join(" | ")} |`);
        }
      }
      L.push("");
    }
  }
  if (R.upbitPath?.events) {
    const u = R.upbitPath;
    L.push(`## 업비트 원화 상장 전후 시간 단위 가격 (바이낸스 1시간봉, BTC 대비, ${u.events}건)`);
    L.push("기준은 업비트에서 첫 거래가 시작된 1시간봉(0시간). 값은 첫 거래 25시간 전 가격 대비 누적 등락.");
    L.push("");
    L.push(`| 시점 | ${u.cumulative.map((c) => (c.hour === 0 ? "첫 거래 시간" : `${c.hour > 0 ? "+" : ""}${c.hour}h`)).join(" | ")} |`);
    L.push(`|---|${u.cumulative.map(() => "---").join("|")}|`);
    L.push(`| 평균 | ${u.cumulative.map((c) => pc(c.mean)).join(" | ")} |`);
    L.push(`| 중앙값 | ${u.cumulative.map((c) => pc(c.median)).join(" | ")} |`);
    L.push("");
    L.push("| 매수 시점 | 24시간 뒤 (평균 / 중앙값 / 오른 비율) | 72시간 뒤 |");
    L.push("|---|---|---|");
    const lab = { h0: "첫 거래 시간봉이 끝났을 때", h1: "그 1시간 뒤", h4: "그 4시간 뒤" };
    for (const b of u.buyAfter) L.push(`| ${lab[b.when]} | ${pc(b.h24.mean)} / ${pc(b.h24.median)} / ${Math.round(b.h24.up * 100)}% | ${pc(b.h72.mean)} / ${pc(b.h72.median)} / ${Math.round(b.h72.up * 100)}% |`);
    L.push("");
    if (R.upbitFast) {
      const f = R.upbitFast;
      L.push(`**1시간 안에 반응했다면** — 첫 거래 전 8시간 안에 1시간 동안 BTC보다 5%p 넘게 뛴 첫 시간봉(공지 시점으로 추정)이 ${f.total}건 중 ${f.found}건에서 잡혔다 (첫 거래의 중앙값 ${f.medianLeadHours}시간 전).`);
      L.push(`- 그 한 시간 동안 이미 오른 폭: 평균 ${pc(f.jumpHour.mean)}, 중앙값 ${pc(f.jumpHour.median)}`);
      L.push(`- 그 시간봉이 끝날 때 사서 업비트 첫 거래 시간봉이 끝날 때 팔면: 평균 ${pc(f.toOpen.mean)}, 중앙값 ${pc(f.toOpen.median)}, 오른 비율 ${Math.round(f.toOpen.up * 100)}% (${f.toOpen.n}건)`);
      L.push(`- 같은 시점에 사서 24시간 들고 있으면: 평균 ${pc(f.to24h.mean)}, 중앙값 ${pc(f.to24h.median)}, 오른 비율 ${Math.round(f.to24h.up * 100)}%`);
      L.push("");
    }
  }
  if (R.h4?.length) {
    L.push("## 메이저 코인 4시간봉 타이밍 규칙 (바이낸스, 수수료 한쪽 0.1%)");
    L.push("규칙의 숫자는 봉 개수 기준이다 (50봉선 ≈ 8일, 30봉 박스 = 5일, 최대 보유 20봉 ≈ 3일).");
    for (const period of ["train", "test"]) {
      L.push("");
      L.push(`**${period === "train" ? "고르는 기간 (2019~2023)" : "확인 기간 (2024~)"}** — 연수익 / 최대 낙폭 · 거래 수 · 승률`);
      L.push("");
      const syms = MAJORS.filter((s) => R.h4.some((x) => x.sym === s));
      L.push(`| 규칙 | 손절 | ${syms.join(" | ")} |`);
      L.push(`|---|---|${syms.map(() => "---").join("|")}|`);
      for (const rule of TIMING_RULES) {
        for (const stop of [0, 0.05]) {
          const rows = syms.map((s) => R.h4.find((x) => x.sym === s && x.rule === rule.id && x.stop === stop));
          if (rows.every((r) => !r)) continue;
          L.push(`| ${rule.label} | ${stop ? "5%" : "없음"} | ${rows.map((r) => (r?.[period] ? `${pc(r[period].cagr)} / ${pc(r[period].mdd)} · ${r[period].trades}회 · ${Math.round(r[period].winRate * 100)}%` : "-")).join(" | ")} |`);
        }
      }
    }
    L.push("");
  }
  L.push("## 주의");
  L.push("- 상장 사건은 \"첫 거래가 시작된 시각\" 기준이다. 상장 공지는 보통 그보다 몇 시간~며칠 앞서 나오고, 가격은 공지 직후 움직인다. 그래서 '공개된 날' 이전 칸에 이미 상승이 잡힌다.");
  L.push("- 해킹은 DefiLlama 목록의 프로토콜 이름이 코인 영문명과 맞는 것만 잡았다. 빠진 사례가 많다.");
  L.push("- 업비트·빗썸은 지금 상장된 종목만 알 수 있다. 상장폐지된 종목의 상장 사건은 빠져 있다.");
  L.push("- 사례 수가 적은 칸은 우연일 수 있다. t값과 사례 수를 같이 본다.");
  return L.join("\n") + "\n";
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
