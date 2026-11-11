#!/usr/bin/env node
/**
 * 검증 실험실 2단계 — 모든 매수 규칙 × 매도 규칙 × 시장 국면 필터를 돌리고,
 * 같은 조건의 무작위 매매 수십 번과 비교해 "우연보다 나은가"를 판정한다.
 *
 *  - 고르는 기간(학습): 데이터 시작 ~ 2023-12-31
 *  - 확인하는 기간(검증): 2024-01-01 ~ 마지막 날
 *  규칙에는 맞춘 숫자가 없으므로 한 번 쭉 돌리고 기간별로 잘라 본다.
 *
 * 결과: lab/results/latest.json, lab/results/REPORT.md
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { loadUniverse } from "./data.mjs";
import { DEFAULTS, holdStats, periodStats, simulate, yearly } from "./engine.mjs";
import { EXITS, STRATEGIES, randomStrategy } from "./strategies.mjs";

const DATA = process.env.LAB_DATA ?? "lab/.data";
const OUT = process.env.LAB_OUT ?? "lab/results";
const SPLIT = process.env.LAB_SPLIT ?? "2024-01-01";
const MIN_VALUE = Number(process.env.LAB_MIN_VALUE ?? 1e9); // 20일 평균 거래대금 10억 원 이상만
const SEEDS = Number(process.env.LAB_SEEDS ?? 30);
const MIN_UNIVERSE = 20;

const pctOf = (x, arr) => (arr.length ? arr.filter((v) => v < x).length / arr.length : NaN);
const r2 = (x) => Math.round(x * 100) / 100;
const r4 = (x) => Math.round(x * 10000) / 10000;

async function main() {
  const t0 = Date.now();
  const U = await loadUniverse(DATA, { minValue: MIN_VALUE });
  U.live = Array.from({ length: U.T }, (_, t) => {
    const out = [];
    for (const k of U.coins) {
      const i = k.idx[t];
      if (i >= 0 && k.eligible[i]) out.push([k, i]);
    }
    return out;
  });

  const start = U.universeSize.findIndex((n) => n >= MIN_UNIVERSE);
  if (start < 0) throw new Error("거래 가능한 종목이 충분한 날이 없습니다");
  const end = U.T - 1;
  const split = U.dates.findIndex((d) => d >= SPLIT);
  if (split <= start || split > end) throw new Error(`분할 날짜 ${SPLIT} 가 데이터 범위 밖입니다`);
  const P = { train: [start, split - 1], test: [split, end] };
  console.log(`종목 ${U.coins.length}개 · 기간 ${U.dates[start]} ~ ${U.dates[end]} · 학습 ${split - start}일 · 검증 ${end - split + 1}일`);

  const evalRun = (run) => ({
    train: periodStats(run, ...P.train),
    test: periodStats(run, ...P.test),
    years: yearly(run, U.dates),
  });

  const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

  const rows = [];
  for (const st of STRATEGIES) {
    for (const exit of EXITS) {
      for (const regime of [false, true]) {
        const key = `${exit.id}${regime ? "+BTC" : ""}`;
        const run = simulate(U, st, exit, { from: start, to: end, regime });
        const ev = evalRun(run);
        // 같은 날 같은 개수를 무작위 종목으로 산 대조군 → 종목 선택력만 비교
        const picks = new Int32Array(U.T);
        for (const x of run.trades) picks[x.signalT]++;
        const rnd = [];
        for (let s = 1; s <= SEEDS; s++) rnd.push(evalRun(simulate(U, randomStrategy(s * 7919), exit, { from: start, to: end, regime, maxPicks: picks })));
        const trainPct = pctOf(ev.train.avgTrade, rnd.map((r) => r.train.avgTrade));
        const testPct = pctOf(ev.test.avgTrade, rnd.map((r) => r.test.avgTrade));
        // 유효 판정: 고르는 기간에 무작위의 90% 이상, 확인 기간에도 80% 이상을 이기고,
        // 확인 기간 거래 30건 이상, 비용 뒤 평균 거래 수익과 연수익이 모두 플러스
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
    console.log(`${st.id} 완료`);
  }

  rows.sort((a, b) => Number(b.valid) - Number(a.valid) || b.testPct - a.testPct || b.test.sharpe - a.test.sharpe);

  const btc = { train: holdStats(U, ...P.train), test: holdStats(U, ...P.test) };
  const meta = await readFile(`${DATA}/meta.json`, "utf8").then(JSON.parse).catch(() => null);
  const result = {
    generatedAt: new Date().toISOString(),
    settings: { ...DEFAULTS, minValue: MIN_VALUE, split: SPLIT, seeds: SEEDS, exchange: "빗썸 원화마켓 일봉" },
    periods: { train: [U.dates[start], U.dates[split - 1]], test: [U.dates[split], U.dates[end]] },
    universe: {
      coins: U.coins.length,
      avgEligibleTrain: Math.round(avg(U.universeSize.slice(...[start, split]))),
      avgEligibleTest: Math.round(avg(U.universeSize.slice(split, end + 1))),
    },
    probes: meta?.probes ?? [],
    btc,
    rows,
  };
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

function report(R) {
  const L = [];
  L.push(`# 지표 검증 결과`);
  L.push("");
  L.push(`생성 ${R.generatedAt.slice(0, 16).replace("T", " ")} UTC · ${R.settings.exchange} · 종목 ${R.universe.coins}개`);
  L.push(`고르는 기간 ${R.periods.train.join(" ~ ")} · 확인 기간 ${R.periods.test.join(" ~ ")}`);
  L.push(`비용: 수수료 한쪽 ${(R.settings.fee * 100).toFixed(2)}% + 슬리피지 ${(R.settings.slipLiquid * 100).toFixed(2)}~${(R.settings.slipThin * 100).toFixed(2)}% · 동시 보유 ${R.settings.slots}종목 · 20일 평균 거래대금 ${R.settings.minValue / 1e8}억 원 이상만`);
  L.push("");
  L.push(`BTC 그냥 보유: 고르는 기간 연 ${pc(R.btc.train.cagr)} (최대 낙폭 ${pc(R.btc.train.mdd)}), 확인 기간 연 ${pc(R.btc.test.cagr)} (최대 낙폭 ${pc(R.btc.test.mdd)})`);
  L.push("");
  const valid = R.rows.filter((r) => r.valid);
  L.push(`## 유효 판정 ${valid.length}개`);
  L.push("고르는 기간에 무작위 대조군의 90% 이상, 확인 기간에도 80% 이상을 이기고, 확인 기간 거래 30건 이상·평균 거래 수익과 연수익이 비용 뒤에도 플러스인 조합.");
  L.push("");
  const head = "| 조합 | 매도 | BTC필터 | 무작위 대비 (고름/확인) | 확인 기간 연수익 | 최대 낙폭 | 거래 | 승률 | 평균 거래 | 무작위 평균 거래 |";
  const sep = "|---|---|---|---|---|---|---|---|---|---|";
  const line = (r) =>
    `| ${r.label} | ${r.exit} | ${r.regime ? "O" : ""} | ${pp(r.trainPct)} / ${pp(r.testPct)} | ${pc(r.test.cagr)} | ${pc(r.test.mdd)} | ${r.test.trades} | ${Math.round(r.test.winRate * 100)}% | ${pc(r.test.avgTrade)} | ${pc(r.random.testAvgTrade)} |`;
  if (valid.length) {
    L.push(head, sep, ...valid.map(line));
  } else L.push("없음");
  L.push("");
  L.push("## 전체 (확인 기간 무작위 대비 순)");
  L.push(head, sep, ...R.rows.map(line));
  L.push("");
  L.push("## 읽을 때 주의");
  L.push("- 지금 빗썸에 상장된 종목만 들어 있다. 상장폐지된 종목이 빠져 있어 실제보다 좋게 나온다 (생존 편향).");
  L.push(`- 무작위 대조군은 그 규칙이 산 날, 산 개수만큼 아무 코인이나 산 경우다 (${R.settings.seeds}번). 숫자는 그중 몇 %보다 평균 거래 수익이 좋았는지이고, 50 근처면 우연과 구별이 안 된다.`);
  L.push("- 변동성이 큰 코인만 고르는 규칙은 오를 때 크게 오르는 효과로 좋아 보일 수 있다. 최대 낙폭을 같이 본다.");
  L.push("- 조합을 많이 돌릴수록 우연히 좋아 보이는 것이 생긴다. 확인 기간에서도 버틴 것만 믿는다.");
  L.push("");
  L.push("## 접속 시험");
  for (const p of R.probes) L.push(`- ${p.status || "실패"} ${p.url}`);
  return L.join("\n") + "\n";
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
