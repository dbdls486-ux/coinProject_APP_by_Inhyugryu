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
import { MAJORS, TIMING_RULES, majorsTiming } from "./timing.mjs";

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
  const { events, note } = JSON.parse(await readFile(`${DATA}/events.json`, "utf8"));
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

    result.markets.push({ market: M.id, label: M.label, periods: { train: [U.dates[P.start], U.dates[P.split - 1]], test: [U.dates[P.split], U.dates[P.end]] }, eventRows, eventList, reactions, listing, pumpRows, pumps, regimeRows, majors });
    console.log(`[${M.label}] 사건 실험 완료`);
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
