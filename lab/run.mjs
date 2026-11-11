#!/usr/bin/env node
/**
 * 검증 실험실 2단계 — 모든 매수 규칙 × 매도 규칙 × 시장 국면 필터를 돌리고,
 * 같은 날 같은 개수를 무작위로 산 대조군과 비교해 "종목을 고르는 힘"이 있는지 판정한다.
 * 빗썸 원화마켓과 바이낸스 USDT 마켓을 따로 돌린다. 두 시장 모두에서 버티는 규칙이 가장 믿을 만하다.
 *
 *  - 고르는 기간(학습): 데이터 시작 ~ 2023-12-31
 *  - 확인하는 기간(검증): 2024-01-01 ~ 마지막 날
 *  규칙에는 맞춘 숫자가 없으므로 한 번 쭉 돌리고 기간별로 잘라 본다.
 *
 * 결과: lab/results/latest.json, lab/results/REPORT.md
 */
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { loadUniverse } from "./data.mjs";
import { DEFAULTS, holdStats, periodStats, simulate, yearly } from "./engine.mjs";
import { EXITS, STRATEGIES, randomStrategy } from "./strategies.mjs";

const DATA = process.env.LAB_DATA ?? "lab/.data";
const OUT = process.env.LAB_OUT ?? "lab/results";
const SPLIT = process.env.LAB_SPLIT ?? "2024-01-01";
const SEEDS = Number(process.env.LAB_SEEDS ?? 30);
const MIN_UNIVERSE = 20;

/** 거래소별 비용·유동성 기준 */
const MARKETS = [
  {
    id: "bithumb",
    label: "빗썸 원화마켓",
    unit: "원",
    minValue: 1e9, // 20일 평균 거래대금 10억 원 이상
    sim: { fee: 0.0025, liquidValue: 1e10, slipLiquid: 0.0015, slipThin: 0.003 },
  },
  {
    id: "binance",
    label: "바이낸스 USDT마켓",
    unit: "USDT",
    minValue: 1e6, // 20일 평균 거래대금 100만 USDT 이상
    sim: { fee: 0.001, liquidValue: 2e7, slipLiquid: 0.0005, slipThin: 0.0015 },
  },
];

const pctOf = (x, arr) => (arr.length ? arr.filter((v) => v < x).length / arr.length : NaN);
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const r4 = (x) => Math.round(x * 10000) / 10000;

async function runMarket(M) {
  const t0 = Date.now();
  const U = await loadUniverse(DATA, { exchange: M.id, minValue: M.minValue });
  U.live = Array.from({ length: U.T }, (_, t) => {
    const out = [];
    for (const k of U.coins) {
      const i = k.idx[t];
      if (i >= 0 && k.eligible[i]) out.push([k, i]);
    }
    return out;
  });

  const start = U.universeSize.findIndex((n) => n >= MIN_UNIVERSE);
  if (start < 0) throw new Error(`${M.label}: 거래 가능한 종목이 ${MIN_UNIVERSE}개 넘는 날이 없습니다`);
  const end = U.T - 1;
  let split = U.dates.findIndex((d) => d >= SPLIT);
  let splitNote = null;
  if (split <= start + 180 || split > end - 90) {
    split = start + Math.floor((end - start) * 0.6);
    splitNote = `데이터 기간이 짧아 ${SPLIT} 대신 ${U.dates[split]} 로 나눴다`;
  }
  const P = { train: [start, split - 1], test: [split, end] };
  console.log(`[${M.label}] 종목 ${U.coins.length}개 · ${U.dates[start]} ~ ${U.dates[end]} · 고르는 기간 ${split - start}일 · 확인 기간 ${end - split + 1}일`);

  const evalRun = (run) => ({
    train: periodStats(run, ...P.train),
    test: periodStats(run, ...P.test),
    years: yearly(run, U.dates),
  });

  const rows = [];
  for (const st of STRATEGIES) {
    for (const exit of EXITS) {
      for (const regime of [false, true]) {
        const key = `${exit.id}${regime ? "+BTC" : ""}`;
        const opt = { ...M.sim, from: start, to: end, regime };
        const run = simulate(U, st, exit, opt);
        const ev = evalRun(run);
        // 같은 날 같은 개수를 무작위 종목으로 산 대조군 → 종목 선택력만 비교
        const picks = new Int32Array(U.T);
        for (const x of run.trades) picks[x.signalT]++;
        const rnd = [];
        for (let s = 1; s <= SEEDS; s++) rnd.push(evalRun(simulate(U, randomStrategy(s * 7919), exit, { ...opt, maxPicks: picks })));
        const trainPct = pctOf(ev.train.avgTrade, rnd.map((r) => r.train.avgTrade));
        const testPct = pctOf(ev.test.avgTrade, rnd.map((r) => r.test.avgTrade));
        // 유효: 고르는 기간 무작위의 90% 이상 + 확인 기간 80% 이상 + 확인 기간 거래 30건 이상 + 비용 뒤 평균 거래·연수익 플러스
        const valid = trainPct >= 0.9 && testPct >= 0.8 && ev.test.trades >= 30 && ev.test.avgTrade > 0 && ev.test.cagr > 0;
        rows.push({
          id: `${st.id}·${key}`,
          strategy: st.id,
          family: st.family,
          label: st.label,
          exit: exit.label,
          regime,
          trainPct,
          testPct,
          valid,
          random: {
            trainAvgTrade: median(rnd.map((r) => r.train.avgTrade)),
            testAvgTrade: median(rnd.map((r) => r.test.avgTrade)),
            testCagr: median(rnd.map((r) => r.test.cagr)),
          },
          ...ev,
          recent: run.trades.filter((x) => x.entryT >= end - 30).map((x) => ({ sym: x.sym, entry: U.dates[x.entryT], exit: U.dates[x.exitT], ret: r4(x.ret), reason: x.reason })),
        });
      }
    }
    console.log(`  ${st.id} 완료`);
  }
  rows.sort((a, b) => Number(b.valid) - Number(a.valid) || b.testPct - a.testPct || b.test.sharpe - a.test.sharpe);

  console.log(`[${M.label}] ${Math.round((Date.now() - t0) / 1000)}초`);
  return {
    market: M.id,
    label: M.label,
    settings: { ...DEFAULTS, ...M.sim, minValue: M.minValue, unit: M.unit, split: U.dates[split], splitNote, seeds: SEEDS },
    periods: { train: [U.dates[start], U.dates[split - 1]], test: [U.dates[split], U.dates[end]] },
    universe: {
      coins: U.coins.length,
      avgEligibleTrain: Math.round(avg(U.universeSize.slice(start, split))),
      avgEligibleTest: Math.round(avg(U.universeSize.slice(split, end + 1))),
    },
    btc: { train: holdStats(U, ...P.train), test: holdStats(U, ...P.test) },
    rows,
  };
}

async function main() {
  const t0 = Date.now();
  const markets = [];
  const skipped = [];
  for (const M of MARKETS) {
    try {
      await access(`${DATA}/${M.id}`);
      markets.push(await runMarket(M));
    } catch (e) {
      console.log(`${M.label} 건너뜀: ${e.message}`);
      skipped.push({ market: M.id, reason: e.message });
    }
  }
  if (markets.length === 0) throw new Error("돌릴 수 있는 시장이 없습니다");

  // 두 시장 모두에서 유효한 조합
  const validSets = markets.map((m) => new Set(m.rows.filter((r) => r.valid).map((r) => r.id)));
  const both = markets.length > 1 ? [...validSets[0]].filter((id) => validSets.every((s) => s.has(id))) : [];

  const meta = await readFile(`${DATA}/meta.json`, "utf8").then(JSON.parse).catch(() => null);
  const result = { generatedAt: new Date().toISOString(), probes: meta?.probes ?? [], toFormat: meta?.toFormat ?? null, both, skipped, markets };
  await mkdir(OUT, { recursive: true });
  await writeFile(`${OUT}/latest.json`, JSON.stringify(result, roundReplacer, 1));
  await writeFile(`${OUT}/REPORT.md`, report(result));
  console.log(`끝 (${Math.round((Date.now() - t0) / 1000)}초)`);
}

function avg(xs) {
  let s = 0;
  for (const x of xs) s += x;
  return xs.length ? s / xs.length : 0;
}

function roundReplacer(_k, v) {
  return typeof v === "number" && !Number.isInteger(v) ? Math.round(v * 10000) / 10000 : v;
}

const pc = (x) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}%`;
const pp = (x) => (Number.isFinite(x) ? `${Math.round(x * 100)}` : "-");
const money = (v, unit) => (unit === "원" ? `${v / 1e8}억 원` : `${v / 1e6}백만 ${unit}`);

function report(R) {
  const L = [];
  L.push(`# 지표 검증 결과`);
  L.push("");
  L.push(`생성 ${R.generatedAt.slice(0, 16).replace("T", " ")} UTC`);
  L.push("");
  const byId = (m, id) => m.rows.find((r) => r.id === id);
  if (R.markets.length > 1) {
    L.push(`## 두 시장 모두에서 유효 ${R.both.length}개`);
    if (R.both.length) {
      for (const id of R.both) {
        const r = byId(R.markets[0], id);
        L.push(`- **${r.label}** · ${r.exit}${r.regime ? " · BTC 50일선 위일 때만" : ""} — ` + R.markets.map((m) => `${m.label} 확인 기간 연 ${pc(byId(m, id).test.cagr)}`).join(", "));
      }
    } else L.push("없음");
    L.push("");
  }
  const head = "| 조합 | 매도 | BTC필터 | 무작위 대비 (고름/확인) | 확인 기간 연수익 | 최대 낙폭 | 거래 | 승률 | 평균 거래 | 무작위 평균 거래 |";
  const sep = "|---|---|---|---|---|---|---|---|---|---|";
  const line = (r) =>
    `| ${r.label} | ${r.exit} | ${r.regime ? "O" : ""} | ${pp(r.trainPct)} / ${pp(r.testPct)} | ${pc(r.test.cagr)} | ${pc(r.test.mdd)} | ${r.test.trades} | ${Math.round(r.test.winRate * 100)}% | ${pc(r.test.avgTrade)} | ${pc(r.random.testAvgTrade)} |`;
  for (const m of R.markets) {
    const s = m.settings;
    L.push(`## ${m.label}`);
    L.push(`종목 ${m.universe.coins}개 (거래 가능 평균: 고르는 기간 ${m.universe.avgEligibleTrain}개, 확인 기간 ${m.universe.avgEligibleTest}개)`);
    L.push(`고르는 기간 ${m.periods.train.join(" ~ ")} · 확인 기간 ${m.periods.test.join(" ~ ")}${s.splitNote ? ` · ${s.splitNote}` : ""}`);
    L.push(`비용: 수수료 한쪽 ${(s.fee * 100).toFixed(2)}% + 슬리피지 ${(s.slipLiquid * 100).toFixed(2)}~${(s.slipThin * 100).toFixed(2)}% · 동시 보유 ${s.slots}종목 · 20일 평균 거래대금 ${money(s.minValue, s.unit)} 이상만`);
    L.push(`BTC 그냥 보유: 고르는 기간 연 ${pc(m.btc.train.cagr)} (최대 낙폭 ${pc(m.btc.train.mdd)}), 확인 기간 연 ${pc(m.btc.test.cagr)} (최대 낙폭 ${pc(m.btc.test.mdd)})`);
    L.push("");
    const valid = m.rows.filter((r) => r.valid);
    L.push(`### 유효 판정 ${valid.length}개`);
    L.push(valid.length ? [head, sep, ...valid.map(line)].join("\n") : "없음");
    L.push("");
    L.push(`### 전체 84개 (확인 기간 무작위 대비 순)`);
    L.push(head, sep, ...m.rows.map(line));
    L.push("");
  }
  L.push("## 읽는 법과 주의");
  L.push("- 유효 = 고르는 기간에 무작위 대조군의 90% 이상, 확인 기간에도 80% 이상을 이기고, 확인 기간 거래 30건 이상·평균 거래 수익과 연수익이 비용 뒤에도 플러스.");
  L.push(`- 무작위 대조군은 그 규칙이 산 날, 산 개수만큼 아무 코인이나 산 경우다 (${R.markets[0].settings.seeds}번). "무작위 대비" 숫자는 그중 몇 %보다 평균 거래 수익이 좋았는지이고, 50 근처면 우연과 구별이 안 된다.`);
  L.push("- 지금 상장된 종목만 들어 있다. 상장폐지된 종목이 빠져 있어 실제보다 좋게 나온다 (생존 편향).");
  L.push("- 변동성이 큰 코인만 고르는 규칙은 오를 때 크게 오르는 효과로 좋아 보일 수 있다. 최대 낙폭을 같이 본다.");
  L.push("- 조합을 많이 돌릴수록 우연히 좋아 보이는 것이 생긴다. 확인 기간에서도 버티고, 두 시장 모두에서 통하는 것을 가장 믿는다.");
  L.push("");
  L.push("## 수집 기록");
  for (const p of R.probes) L.push(`- 접속 ${p.status || "실패"} ${p.url}`);
  L.push(`- 빗썸 과거 페이지 형식: ${R.toFormat ?? "찾지 못함 (최근 200일만)"}`);
  for (const s of R.skipped) L.push(`- 건너뜀 ${s.market}: ${s.reason}`);
  return L.join("\n") + "\n";
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
