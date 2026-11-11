/**
 * 여러 실행 스크립트가 같이 쓰는 설정과 준비 과정.
 */
import { access } from "node:fs/promises";
import { loadUniverse } from "./data.mjs";
import { attachFunding, attachKimchi } from "./features.mjs";

export const DATA = process.env.LAB_DATA ?? "lab/.data";
export const OUT = process.env.LAB_OUT ?? "lab/results";
export const SPLIT = process.env.LAB_SPLIT ?? "2024-01-01";
export const MIN_UNIVERSE = 20;

export const MARKETS = [
  {
    id: "bithumb",
    label: "빗썸 원화마켓",
    unit: "원",
    tzOffsetH: 9, // 일봉 날짜가 한국 날짜
    dayEndUtcH: 15, // 날짜 D 의 봉은 D 15:00 UTC 에 끝난다
    minValue: 1e9, // 20일 평균 거래대금 10억 원 이상
    sim: { fee: 0.0025, liquidValue: 1e10, slipLiquid: 0.0015, slipThin: 0.003 },
  },
  {
    id: "binance",
    label: "바이낸스 USDT마켓",
    unit: "USDT",
    tzOffsetH: 0,
    dayEndUtcH: 24, // 날짜 D 의 봉은 D+1 00:00 UTC 에 끝난다
    minValue: 1e6, // 20일 평균 거래대금 100만 USDT 이상
    sim: { fee: 0.001, liquidValue: 2e7, slipLiquid: 0.0005, slipThin: 0.0015 },
  },
];

export async function prepare(M, topN) {
  const today = new Date(Date.now() + M.tzOffsetH * 3_600_000).toISOString().slice(0, 10);
  const U = await loadUniverse(DATA, { exchange: M.id, minValue: M.minValue, dropFrom: today, topN });
  U.today = today;
  U.live = Array.from({ length: U.T }, (_, t) => {
    const out = [];
    for (const k of U.coins) {
      const i = k.idx[t];
      if (i >= 0 && k.eligible[i]) out.push([k, i]);
    }
    return out;
  });
  return U;
}

/** 두 시장을 함께 불러 김프·펀딩비까지 붙인다 */
export async function prepareAll(topN) {
  const U = {};
  const skipped = [];
  const extras = {};
  for (const M of MARKETS) {
    try {
      await access(`${DATA}/${M.id}`);
      U[M.id] = await prepare(M, topN);
    } catch (e) {
      skipped.push({ market: M.id, label: M.label, reason: e.message });
    }
  }
  if (U.bithumb && U.binance) extras.kimchi = attachKimchi(U.bithumb, U.binance);
  if (U.binance) extras.fundingBinance = await attachFunding(U.binance, DATA, 0);
  if (U.bithumb) extras.fundingBithumb = await attachFunding(U.bithumb, DATA, 1);
  return { U, skipped, extras };
}

/** 고르는 기간·확인 기간의 날짜 번호 */
export function periods(U) {
  const start = U.universeSize.findIndex((n) => n >= MIN_UNIVERSE);
  if (start < 0) throw new Error(`거래 가능한 종목이 ${MIN_UNIVERSE}개 넘는 날이 없음`);
  const end = U.T - 1;
  const split = U.dates.findIndex((d) => d >= SPLIT);
  if (split - start < 365 || end - split < 180) throw new Error(`데이터 기간이 짧아 판정하지 않음 (${U.dates[start]} ~ ${U.dates[end]})`);
  return { start, split, end, train: [start, split - 1], test: [split, end] };
}
