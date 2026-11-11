#!/usr/bin/env node
/**
 * 검증 실험실 — 2차 설계 (독립 점검 결과 반영)
 *
 * 두 질문을 따로 잰다.
 *  A. 종목 선택력: 규칙이 고른 코인이 같은 날·같은 변동성 구간의 다른 코인보다 나았나 (selection.mjs)
 *     - 고르는 기간과 확인 기간 모두 겹침 보정 t ≥ 2, 로그 수익 기준으로도 플러스여야 통과
 *     - 무작위로 고르는 위약 규칙 10개를 같이 돌려, 판정 기준이 우연을 걸러내는지 확인
 *  B. 시기 선택: 같은 규칙을 시장 국면 필터 5가지로 굴렸을 때 실제 계좌가 어땠나 (engine.mjs, regimes.mjs)
 *     - 비교 기준: BTC 그냥 보유, BTC 50일선 타이밍
 *
 * 고르는 기간: 데이터 시작 ~ 2023-12-31 / 확인 기간: 2024-01-01 ~
 * 결과: lab/results/latest.json, lab/results/REPORT.md
 */
import { access, mkdir, readFile, writeFile } from "node:fs/promises";
import { btcTiming } from "./bench.mjs";
import { loadUniverse } from "./data.mjs";
import { DEFAULTS, holdStats, periodStats, simulate, yearly } from "./engine.mjs";
import { buildRegimes } from "./regimes.mjs";
import { selectionStats, tradeTable } from "./selection.mjs";
import { EXITS, STRATEGIES, randomStrategy } from "./strategies.mjs";

const DATA = process.env.LAB_DATA ?? "lab/.data";
const OUT = process.env.LAB_OUT ?? "lab/results";
const SPLIT = process.env.LAB_SPLIT ?? "2024-01-01";
const PLACEBOS = Number(process.env.LAB_PLACEBOS ?? 10);
const MIN_UNIVERSE = 20;
const T_PASS = 2;

const MARKETS = [
  {
    id: "bithumb",
    label: "빗썸 원화마켓",
    unit: "원",
    tzOffsetH: 9, // 일봉 날짜가 한국 날짜
    minValue: 1e9, // 20일 평균 거래대금 10억 원 이상
    sim: { fee: 0.0025, liquidValue: 1e10, slipLiquid: 0.0015, slipThin: 0.003 },
  },
  {
    id: "binance",
    label: "바이낸스 USDT마켓",
    unit: "USDT",
    tzOffsetH: 0,
    minValue: 1e6, // 20일 평균 거래대금 100만 USDT 이상
    sim: { fee: 0.001, liquidValue: 2e7, slipLiquid: 0.0005, slipThin: 0.0015 },
  },
];

/** 무작위 위약 규칙: 매일 약 10% 종목에 무작위 점수 */
function placebo(seed) {
  const r = randomStrategy(seed);
  return { id: `PLACEBO${seed}`, family: "위약", label: `무작위 선택 ${seed}`, entry: (k, i, t) => (r.entry() < 0.1 ? r.entry() : null) };
}

async function runMarket(M) {
  const t0 = Date.now();
  const today = new Date(Date.now() + M.tzOffsetH * 3_600_000).toISOString().slice(0, 10);
  const U = await loadUniverse(DATA, { exchange: M.id, minValue: M.minValue, dropFrom: today });
  U.live = Array.from({ length: U.T }, (_, t) => {
    const out = [];
    for (const k of U.coins) {
      const i = k.idx[t];
      if (i >= 0 && k.eligible[i]) out.push([k, i]);
    }
    return out;
  });
  const start = U.universeSize.findIndex((n) => n >= MIN_UNIVERSE);
  if (start < 0) throw new Error(`거래 가능한 종목이 ${MIN_UNIVERSE}개 넘는 날이 없음`);
  const end = U.T - 1;
  const split = U.dates.findIndex((d) => d >= SPLIT);
  if (split - start < 365 || end - split < 180) throw new Error(`데이터 기간이 짧아 판정하지 않음 (${U.dates[start]} ~ ${U.dates[end]})`);
  const P = { train: [start, split - 1], test: [split, end] };
  console.log(`[${M.label}] 종목 ${U.coins.length}개 · ${U.dates[start]} ~ ${U.dates[end]} · 액면 변경 의심으로 앞부분을 자른 종목 ${U.truncated}개`);

  const regimes = buildRegimes(U);

  // ---------- A. 종목 선택력 ----------
  const selection = [];
  const placebos = [];
  for (const exit of EXITS) {
    const T = tradeTable(U, exit, M.sim);
    const trainTo = split - 2 - exit.maxHold; // 고르는 기간 거래가 확인 기간으로 넘어가지 않게
    const evalSel = (st) => {
      const train = selectionStats(U, st, T, start, trainTo);
      const test = selectionStats(U, st, T, split, end);
      const pass = train.t >= T_PASS && test.t >= T_PASS && train.logExcess > 0 && test.logExcess > 0;
      return { strategy: st.id, family: st.family, label: st.label, exit: exit.id, exitLabel: exit.label, train, test, pass };
    };
    for (const st of STRATEGIES) selection.push(evalSel(st));
    for (let s = 1; s <= PLACEBOS; s++) placebos.push(evalSel(placebo(s * 104729 + exit.maxHold)));
  }
  console.log(`  선택력 평가 완료`);

  // ---------- B. 국면 필터별 실제 계좌 ----------
  const fee = M.sim.fee;
  const bench = {
    hold: { train: holdStats(U, ...P.train), test: holdStats(U, ...P.test) },
    timing: { train: btcTiming(U, ...P.train, fee), test: btcTiming(U, ...P.test, fee) },
  };
  const portfolio = [];
  for (const st of STRATEGIES) {
    for (const exit of EXITS) {
      for (const rg of regimes) {
        const run = simulate(U, st, exit, { ...M.sim, from: start, to: end, regimeOn: rg.on });
        portfolio.push({
          strategy: st.id,
          exit: exit.id,
          regime: rg.id,
          train: periodStats(run, ...P.train),
          test: periodStats(run, ...P.test),
          years: yearly(run, U.dates),
        });
      }
    }
  }
  console.log(`  국면별 계좌 시뮬레이션 완료`);

  // 국면 필터별 요약: 모든 규칙×매도 조합의 중앙값
  const med = (xs) => {
    const a = [...xs].sort((x, y) => x - y);
    return a.length ? a[Math.floor(a.length / 2)] : NaN;
  };
  const regimeSummary = regimes.map((rg) => {
    const rs = portfolio.filter((p) => p.regime === rg.id);
    const onDays = (a, b) => {
      let n = 0;
      for (let t = a; t <= b; t++) n += rg.on[t];
      return n / (b - a + 1);
    };
    return {
      regime: rg.id,
      label: rg.label,
      onTrain: onDays(...P.train),
      onTest: onDays(...P.test),
      trainCagr: med(rs.map((r) => r.train.cagr)),
      testCagr: med(rs.map((r) => r.test.cagr)),
      testMdd: med(rs.map((r) => r.test.mdd)),
      testPositive: rs.filter((r) => r.test.cagr > 0).length / rs.length,
      beatTimingTest: rs.filter((r) => r.test.sharpe > bench.timing.test.sharpe).length / rs.length,
    };
  });

  // 실전 후보: 선택력 통과 + 고르는 기간 샤프가 가장 좋은 국면 필터로 확인 기간에도 플러스
  const candidates = [];
  for (const s of selection) {
    const opts = portfolio.filter((p) => p.strategy === s.strategy && p.exit === s.exit);
    const best = opts.reduce((a, b) => (b.train.sharpe > a.train.sharpe ? b : a));
    const ok = s.pass && best.test.cagr > 0;
    candidates.push({ strategy: s.strategy, label: s.label, exit: s.exit, exitLabel: s.exitLabel, selectionPass: s.pass, regime: best.regime, train: best.train, test: best.test, years: best.years, candidate: ok, beatsTiming: best.test.sharpe > bench.timing.test.sharpe });
  }

  console.log(`[${M.label}] ${Math.round((Date.now() - t0) / 1000)}초`);
  return {
    market: M.id,
    label: M.label,
    settings: { ...DEFAULTS, ...M.sim, minValue: M.minValue, unit: M.unit, split: U.dates[split], tPass: T_PASS, placebos: PLACEBOS, droppedFrom: today },
    periods: { train: [U.dates[start], U.dates[split - 1]], test: [U.dates[split], U.dates[end]] },
    universe: {
      coins: U.coins.length,
      truncated: U.truncated,
      avgEligibleTrain: Math.round(avg(U.universeSize.slice(start, split))),
      avgEligibleTest: Math.round(avg(U.universeSize.slice(split, end + 1))),
    },
    bench,
    regimeSummary,
    selection,
    placebo: {
      count: placebos.length,
      passed: placebos.filter((p) => p.pass).length,
      maxTrainT: Math.max(...placebos.map((p) => p.train.t)),
      maxTestT: Math.max(...placebos.map((p) => p.test.t)),
      trainOver2: placebos.filter((p) => p.train.t >= T_PASS).length,
      testOver2: placebos.filter((p) => p.test.t >= T_PASS).length,
    },
    candidates,
    portfolio,
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
      skipped.push({ market: M.id, label: M.label, reason: e.message });
    }
  }
  if (markets.length === 0) throw new Error("돌릴 수 있는 시장이 없습니다");
  const meta = await readFile(`${DATA}/meta.json`, "utf8").then(JSON.parse).catch(() => null);
  const result = { generatedAt: new Date().toISOString(), design: 2, collect: meta ? { probes: meta.probes, toFormat: meta.toFormat, checks: meta.checks ?? null } : null, skipped, markets };
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

const pc = (x) => (Number.isFinite(x) ? `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}%` : "-");
const pc2 = (x) => (Number.isFinite(x) ? `${x >= 0 ? "+" : ""}${(x * 100).toFixed(2)}%p` : "-");
const t1 = (x) => (Number.isFinite(x) ? x.toFixed(1) : "-");
const money = (v, unit) => (unit === "원" ? `${v / 1e8}억 원` : `${v / 1e6}백만 ${unit}`);
const EXIT_LABEL = Object.fromEntries(EXITS.map((e) => [e.id, e.label]));
const STRAT_LABEL = Object.fromEntries(STRATEGIES.map((s) => [s.id, s.label]));

function report(R) {
  const L = [];
  L.push("# 지표 검증 결과");
  L.push("");
  L.push(`생성 ${R.generatedAt.slice(0, 16).replace("T", " ")} UTC · 2차 설계 (같은 날·같은 변동성 비교, 위약 규칙, 국면 필터 5종)`);
  L.push("");

  // 결론
  L.push("## 한눈에");
  for (const m of R.markets) {
    const pass = m.selection.filter((s) => s.pass);
    const cand = m.candidates.filter((c) => c.candidate);
    const bestRegime = [...m.regimeSummary].sort((a, b) => b.testCagr - a.testCagr)[0];
    L.push(`- **${m.label}**: 종목 선택력 통과 ${pass.length}개 / 42개 (위약 ${m.placebo.passed}개 / ${m.placebo.count}개 통과) · 실전 후보 ${cand.length}개 · 국면 필터 중 확인 기간 성적이 가장 나은 것은 "${bestRegime.label}" (규칙 중앙값 연 ${pc(bestRegime.testCagr)}) · BTC 보유 연 ${pc(m.bench.hold.test.cagr)}, BTC 50일선 타이밍 연 ${pc(m.bench.timing.test.cagr)}`);
  }
  for (const s of R.skipped) L.push(`- ${s.label}: 판정 안 함 — ${s.reason}`);
  L.push("");

  for (const m of R.markets) {
    const s = m.settings;
    L.push(`## ${m.label}`);
    L.push(`종목 ${m.universe.coins}개 (거래 가능 평균: 고르는 기간 ${m.universe.avgEligibleTrain}개, 확인 기간 ${m.universe.avgEligibleTest}개) · 액면 변경 의심으로 앞부분을 자른 종목 ${m.universe.truncated}개`);
    L.push(`고르는 기간 ${m.periods.train.join(" ~ ")} · 확인 기간 ${m.periods.test.join(" ~ ")} · ${s.droppedFrom} 미완성 봉 제외`);
    L.push(`비용: 수수료 한쪽 ${(s.fee * 100).toFixed(2)}% + 슬리피지 ${(s.slipLiquid * 100).toFixed(2)}~${(s.slipThin * 100).toFixed(2)}% (손절은 2배) · 20일 평균 거래대금 ${money(s.minValue, s.unit)} 이상만 · 동시 보유 ${s.slots}종목`);
    L.push("");

    L.push("### 1. 시장 국면 필터 비교 (규칙 42개 조합의 중앙값)");
    L.push("| 필터 | 매수 허용 날 (고름/확인) | 고르는 기간 연수익 | 확인 기간 연수익 | 확인 기간 최대 낙폭 | 확인 기간 플러스 비율 | BTC 타이밍보다 나은 비율 |");
    L.push("|---|---|---|---|---|---|---|");
    for (const r of m.regimeSummary) L.push(`| ${r.label} | ${Math.round(r.onTrain * 100)}% / ${Math.round(r.onTest * 100)}% | ${pc(r.trainCagr)} | ${pc(r.testCagr)} | ${pc(r.testMdd)} | ${Math.round(r.testPositive * 100)}% | ${Math.round(r.beatTimingTest * 100)}% |`);
    L.push(`| *BTC 그냥 보유* | | ${pc(m.bench.hold.train.cagr)} | ${pc(m.bench.hold.test.cagr)} | ${pc(m.bench.hold.test.mdd)} | | |`);
    L.push(`| *BTC 50일선 타이밍* | | ${pc(m.bench.timing.train.cagr)} | ${pc(m.bench.timing.test.cagr)} | ${pc(m.bench.timing.test.mdd)} | | |`);
    L.push("");

    L.push("### 2. 종목 선택력 (같은 날·같은 변동성 구간 평균 대비 거래당 초과수익)");
    L.push(`통과 = 고르는 기간·확인 기간 모두 t ≥ ${s.tPass} 이고 로그 수익 기준으로도 플러스. 위약 규칙 ${m.placebo.count}개 중 고르는 기간 t≥${s.tPass} ${m.placebo.trainOver2}개, 확인 기간 ${m.placebo.testOver2}개, 둘 다 통과 ${m.placebo.passed}개 (최대 t ${t1(m.placebo.maxTrainT)} / ${t1(m.placebo.maxTestT)}).`);
    L.push("");
    L.push("| 규칙 | 매도 | 고르는 기간 초과 (t) | 확인 기간 초과 (t) | 확인 기간 로그 초과 | 확인 기간 고른 횟수 | 통과 |");
    L.push("|---|---|---|---|---|---|---|");
    const sel = [...m.selection].sort((a, b) => Number(b.pass) - Number(a.pass) || Math.min(b.train.t, b.test.t) - Math.min(a.train.t, a.test.t));
    for (const r of sel) L.push(`| ${r.label} | ${r.exitLabel} | ${pc2(r.train.excess)} (${t1(r.train.t)}) | ${pc2(r.test.excess)} (${t1(r.test.t)}) | ${pc2(r.test.logExcess)} | ${r.test.picks} | ${r.pass ? "O" : ""} |`);
    L.push("");

    L.push("### 3. 규칙별 최적 국면 필터로 굴린 계좌 (필터는 고르는 기간 샤프로 선택)");
    L.push("| 규칙 | 매도 | 필터 | 고르는 기간 연수익 | 확인 기간 연수익 | 확인 기간 최대 낙폭 | 확인 기간 샤프 | 선택력 | 후보 |");
    L.push("|---|---|---|---|---|---|---|---|---|");
    const rgLabel = Object.fromEntries(m.regimeSummary.map((r) => [r.regime, r.label]));
    const cands = [...m.candidates].sort((a, b) => Number(b.candidate) - Number(a.candidate) || b.test.sharpe - a.test.sharpe);
    for (const c of cands) L.push(`| ${c.label} | ${c.exitLabel} | ${rgLabel[c.regime]} | ${pc(c.train.cagr)} | ${pc(c.test.cagr)} | ${pc(c.test.mdd)} | ${c.test.sharpe.toFixed(2)} | ${c.selectionPass ? "O" : ""} | ${c.candidate ? "O" : ""} |`);
    L.push(`비교: BTC 50일선 타이밍 확인 기간 샤프 ${m.bench.timing.test.sharpe.toFixed(2)}, BTC 보유 ${m.bench.hold.test.sharpe.toFixed(2)}`);
    L.push("");

    L.push("### 4. 연도별 수익 (확인 기간 샤프 상위 6개 조합)");
    const years = Object.keys(cands[0]?.years ?? {}).sort();
    L.push(`| 규칙·매도·필터 | ${years.join(" | ")} |`);
    L.push(`|---|${years.map(() => "---").join("|")}|`);
    for (const c of cands.slice(0, 6)) L.push(`| ${STRAT_LABEL[c.strategy]} · ${EXIT_LABEL[c.exit]} · ${rgLabel[c.regime]} | ${years.map((y) => pc(c.years[y])).join(" | ")} |`);
    L.push("");
  }

  L.push("## 읽는 법");
  L.push("- **종목 선택력**: 규칙이 고른 코인의 거래 수익에서, 같은 날 같은 변동성 5분위에 있던 모든 코인을 같은 매도 규칙으로 샀을 때의 평균을 뺀 값. 무작위 표본이 아니라 전수 평균이라 매수 시점·개수·변동성이 정확히 같다.");
  L.push("- **t**: 보유 기간이 겹치는 것을 보정한 값. 2 이상이면 우연일 가능성이 낮다고 본다. 위약 규칙이 거의 통과하지 못해야 기준이 제대로 작동하는 것이다.");
  L.push("- **국면 필터**: 필터가 꺼진 날에는 새로 사지 않는다 (들고 있는 것은 원래 매도 규칙대로).");
  L.push("- **후보** = 선택력 통과 + 고르는 기간에 고른 국면 필터로 확인 기간 연수익도 플러스.");
  L.push("- 지금 상장된 종목만 들어 있다. 상장폐지 종목이 빠져 있어 실제보다 좋게 나온다 (생존 편향). 특히 낙폭 과대·거래량 급증 규칙이 유리해진다.");
  L.push("- 빗썸과 바이낸스는 같은 코인이 많고 가격이 같이 움직여, 두 시장에서 같은 결과가 나와도 완전히 독립된 확인은 아니다.");
  L.push("");
  if (R.collect) {
    L.push("## 수집 기록");
    for (const p of R.collect.probes ?? []) L.push(`- 접속 ${p.status || "실패"} ${p.url}`);
    L.push(`- 빗썸 과거 페이지 형식: ${R.collect.toFormat ?? "찾지 못함"}`);
    if (R.collect.checks) L.push(`- 점검: ${JSON.stringify(R.collect.checks)}`);
  }
  return L.join("\n") + "\n";
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
