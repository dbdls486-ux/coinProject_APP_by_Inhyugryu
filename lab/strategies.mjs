/**
 * 검증할 매수 규칙 후보. 각 규칙은 (종목 k, 그 종목의 봉 i, 날짜 t) 를 받아
 * 오늘 종가 기준으로 매수 후보면 점수(클수록 우선), 아니면 null 을 돌려준다.
 * 실제 매수는 다음 날 시가에 한다 (엔진이 처리).
 *
 * 근거 (README 참고):
 *  - 모멘텀: 1~4주 수익률 상위가 계속 오른다 (Liu·Tsyvinski·Wu 2022, Dobrynskaya — 2주가 가장 강하고 4~6주부터 반전)
 *  - 추세 추종: 이동평균·돈치안 돌파 (Detzel 외 2021, Barbon 2025)
 *  - MAX 효과: 최근 한 달 최대 일간 상승률이 큰 코인이 이후에도 강하다 (Özdamar 외 2021)
 *  - 단기 역추세·눌림목: 코인에서는 약하다는 결과가 많지만 대조군으로 넣는다
 */

const ok = (x) => Number.isFinite(x);

export const STRATEGIES = [
  // ---------- 모멘텀 ----------
  {
    id: "MOM7",
    family: "모멘텀",
    label: "7일 상승률 상위 10%",
    entry: (k, i) => (k.p.ret7[i] >= 0.9 ? k.p.ret7[i] : null),
  },
  {
    id: "MOM14",
    family: "모멘텀",
    label: "14일 상승률 상위 10%",
    entry: (k, i) => (k.p.ret14[i] >= 0.9 ? k.p.ret14[i] : null),
  },
  {
    id: "MOM28",
    family: "모멘텀",
    label: "28일 상승률 상위 10%",
    entry: (k, i) => (k.p.ret28[i] >= 0.9 ? k.p.ret28[i] : null),
  },
  {
    id: "RS14",
    family: "모멘텀",
    label: "BTC보다 14일간 강한 상위 20% + 20일선 위",
    entry: (k, i) => (k.p.rs14[i] >= 0.8 && k.c[i] > k.f.sma20[i] ? k.p.rs14[i] : null),
  },
  // ---------- 추세 추종 ----------
  {
    id: "MA_ALIGN",
    family: "추세",
    label: "종가 > 20일선 > 50일선 으로 막 전환",
    entry: (k, i) => {
      const f = k.f;
      if (i < 1 || !ok(f.sma50[i - 1])) return null;
      const now = k.c[i] > f.sma20[i] && f.sma20[i] > f.sma50[i];
      const before = k.c[i - 1] > f.sma20[i - 1] && f.sma20[i - 1] > f.sma50[i - 1];
      return now && !before ? (ok(f.ret14[i]) ? f.ret14[i] : 0) : null;
    },
  },
  {
    id: "DONCH20",
    family: "추세",
    label: "20일 최고가 돌파",
    entry: (k, i) => (ok(k.f.hh20[i]) && k.c[i] > k.f.hh20[i] ? k.c[i] / k.f.hh20[i] : null),
  },
  {
    id: "DONCH_ENS",
    family: "추세",
    label: "10·20·40·60일 돌파 중 3개 이상으로 막 전환",
    entry: (k, i) => {
      if (i < 1) return null;
      const f = k.f;
      const score = (j) => [f.hh10, f.hh20, f.hh40, f.hh60].reduce((s, hh) => s + (ok(hh[j]) && k.c[j] > hh[j] ? 1 : 0), 0) / 4;
      const now = score(i);
      return now >= 0.75 && score(i - 1) < 0.75 ? now + (ok(f.ret14[i]) ? f.ret14[i] * 0.01 : 0) : null;
    },
  },
  {
    id: "BRK_VOL",
    family: "추세",
    label: "20일 최고가 돌파 + 거래대금 평소 2배 이상",
    entry: (k, i) => (ok(k.f.hh20[i]) && k.c[i] > k.f.hh20[i] && k.f.volRatio[i] >= 2 ? k.f.volRatio[i] : null),
  },
  // ---------- 눌림목·역추세 ----------
  {
    id: "PULLBACK",
    family: "눌림목",
    label: "상승 추세(20일선>50일선, 종가>50일선) 속 RSI 45 미만 조정",
    entry: (k, i) => {
      const f = k.f;
      return f.sma20[i] > f.sma50[i] && k.c[i] > f.sma50[i] && f.rsi14[i] < 45 ? 45 - f.rsi14[i] : null;
    },
  },
  {
    id: "RSI2_UP",
    family: "눌림목",
    label: "100일선 위에서 RSI(2) 10 미만",
    entry: (k, i) => (k.c[i] > k.f.sma100[i] && k.f.rsi2[i] < 10 ? 10 - k.f.rsi2[i] : null),
  },
  {
    id: "BB_LOW",
    family: "역추세",
    label: "볼린저밴드 하단 아래 마감",
    entry: (k, i) => (ok(k.f.bbLow[i]) && k.c[i] < k.f.bbLow[i] ? (k.f.bbLow[i] - k.c[i]) / k.c[i] : null),
  },
  {
    id: "REV7",
    family: "역추세",
    label: "7일 하락률 하위 10% (낙폭 과대)",
    entry: (k, i) => (k.p.ret7[i] <= 0.1 ? 1 - k.p.ret7[i] : null),
  },
  // ---------- 관심·거래량 ----------
  {
    id: "MAX30",
    family: "관심",
    label: "최근 30일 최대 하루 상승률 상위 10%",
    entry: (k, i) => (k.p.max30[i] >= 0.9 ? k.p.max30[i] : null),
  },
  {
    id: "VOL_SURGE",
    family: "관심",
    label: "거래대금 평소 3배 이상 + 양봉",
    entry: (k, i) => (k.f.volRatio[i] >= 3 && k.c[i] > k.o[i] ? k.f.volRatio[i] : null),
  },
];

/** 매도 규칙: 매수가 - 2×ATR 손절 + 최대 보유일 */
export const EXITS = [
  { id: "H7", label: "7일 보유", maxHold: 7, stopAtr: 2 },
  { id: "H14", label: "14일 보유", maxHold: 14, stopAtr: 2 },
  { id: "H28", label: "28일 보유", maxHold: 28, stopAtr: 2 },
];

/**
 * 무작위 대조군: 거래 가능한 종목 중 아무거나.
 * 엔진에 maxPicks 를 함께 주면 원래 규칙이 산 날, 산 개수만큼만 무작위로 산다.
 * → 매수 시점·보유 비중은 같고 "어떤 코인을 골랐나"만 다르다.
 */
export function randomStrategy(seed) {
  let s = seed >>> 0 || 1;
  const rand = () => {
    // mulberry32
    s = (s + 0x6d2b79f5) >>> 0;
    let x = s;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
  return { id: `RANDOM#${seed}`, family: "대조군", label: "무작위", entry: () => rand() };
}
