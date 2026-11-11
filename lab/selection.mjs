/**
 * 종목 선택력 평가 — "이 규칙이 고른 코인이, 같은 날 같은 변동성 구간의 다른 코인보다 나았나".
 *
 * 1) 거래 표: 거래 가능한 모든 (종목, 신호일) 에 대해 같은 매도 규칙으로 혼자 거래했을 때의 순수익을 한 번 계산한다.
 *    다음 날 시가 매수, 매수가 − k×ATR 손절(슬리피지 2배), 최대 보유일 경과 시 시가 매도. 수수료·슬리피지 양쪽 반영.
 * 2) 기준값: 그날 같은 변동성 5분위 안 모든 종목의 평균 수익. 무작위 표본이 아니라 전수 평균이라 시드가 필요 없고,
 *    매수 시점·개수·변동성이 정확히 같다.
 * 3) 규칙이 그날 고른 종목(점수 상위 최대 10개)의 초과수익을 날짜별로 평균 내고,
 *    보유 기간이 겹치므로 Newey-West(시차 = 최대 보유일) 로 보정한 t값을 쓴다.
 */

export function tradeTable(U, exit, sim) {
  const fee = sim.fee;
  const table = new Map();
  for (const k of U.coins) {
    const ret = new Float64Array(k.n).fill(NaN);
    for (let i = 0; i < k.n - 1; i++) {
      if (!k.eligible[i]) continue;
      const a = k.f.atr14[i];
      if (!(a > 0)) continue;
      const slip = k.f.val20[i] >= sim.liquidValue ? sim.slipLiquid : sim.slipThin;
      const entry = k.o[i + 1] * (1 + slip);
      const stop = entry - exit.stopAtr * a;
      const last = i + 1 + exit.maxHold;
      let px = NaN;
      for (let j = i + 1; j < k.n; j++) {
        if (j === last) {
          px = k.o[j] * (1 - slip);
          break;
        }
        if (k.l[j] <= stop) {
          px = Math.min(k.o[j], stop) * (1 - 2 * slip);
          break;
        }
      }
      // 매도 전에 봉이 끝났고 그 종목이 데이터 끝보다 먼저 사라졌다면(상장폐지) 마지막 종가로 정리.
      // 아직 거래 중인데 최근 신호라 끝나지 않은 거래는 제외한다.
      if (!Number.isFinite(px) && k.day[k.n - 1] < U.T - 1 - 3) px = k.c[k.n - 1] * (1 - 2 * slip);
      if (Number.isFinite(px)) ret[i] = ((1 - fee) * (1 - fee) * px) / entry - 1;
    }
    table.set(k, ret);
  }
  // 날짜 × 변동성 5분위 평균 (단순 수익, 로그 수익)
  const mean = new Float64Array(U.T * 5).fill(NaN);
  const meanLog = new Float64Array(U.T * 5).fill(NaN);
  for (let t = 0; t < U.T; t++) {
    const s = [0, 0, 0, 0, 0];
    const sl = [0, 0, 0, 0, 0];
    const n = [0, 0, 0, 0, 0];
    for (const [k, i] of U.live[t]) {
      const r = table.get(k)[i];
      const b = k.bucket[i];
      if (b < 0 || !Number.isFinite(r)) continue;
      s[b] += r;
      sl[b] += Math.log1p(r);
      n[b]++;
    }
    for (let b = 0; b < 5; b++) {
      if (n[b] >= 2) {
        mean[t * 5 + b] = s[b] / n[b];
        meanLog[t * 5 + b] = sl[b] / n[b];
      }
    }
  }
  return { table, mean, meanLog, exit };
}

/** Newey-West 보정 t값 */
function hacT(xs, lag) {
  const n = xs.length;
  if (n < 10) return { mean: n ? xs.reduce((a, b) => a + b, 0) / n : 0, t: 0 };
  const m = xs.reduce((a, b) => a + b, 0) / n;
  const d = xs.map((x) => x - m);
  const gamma = (l) => {
    let s = 0;
    for (let i = l; i < n; i++) s += d[i] * d[i - l];
    return s / n;
  };
  let v = gamma(0);
  const L = Math.min(lag, n - 1);
  for (let l = 1; l <= L; l++) v += 2 * (1 - l / (L + 1)) * gamma(l);
  const se = Math.sqrt(Math.max(v, 1e-18) / n);
  return { mean: m, t: m / se };
}

/**
 * @param from,to 신호일 범위 (날짜 번호). 고르는 기간은 매도까지 끝나도록 to 를 당겨서 넘긴다.
 * @param regimeOn 주어지면 그 날만 본다
 */
export function selectionStats(U, strat, T, from, to, regimeOn = null, perDay = 10, plainT = false) {
  const xs = [];
  const xl = [];
  const each = [];
  let picks = 0;
  let raw = 0;
  for (let t = from; t <= to; t++) {
    if (regimeOn && !regimeOn[t]) continue;
    const cands = [];
    for (const [k, i] of U.live[t]) {
      const b = k.bucket[i];
      if (b < 0) continue;
      const r = T.table.get(k)[i];
      if (!Number.isFinite(r) || !Number.isFinite(T.mean[t * 5 + b])) continue;
      const s = strat.entry(k, i, t);
      if (s === null || !Number.isFinite(s)) continue;
      cands.push([s, r, b]);
    }
    if (cands.length === 0) continue;
    cands.sort((a, b) => b[0] - a[0]);
    const top = cands.slice(0, perDay);
    let e = 0;
    let el = 0;
    for (const [, r, b] of top) {
      e += r - T.mean[t * 5 + b];
      el += Math.log1p(r) - T.meanLog[t * 5 + b];
      raw += r;
      each.push(r - T.mean[t * 5 + b]);
    }
    xs.push(e / top.length);
    xl.push(el / top.length);
    picks += top.length;
  }
  // 드문 사건은 날짜가 띄엄띄엄이라 겹침 보정이 의미가 없다 → 단순 t (plainT)
  const lag = plainT ? 0 : T.exit.maxHold;
  const s = hacT(xs, lag);
  const l = hacT(xl, lag);
  each.sort((a, b) => a - b);
  const medExcess = each.length ? each[Math.floor(each.length / 2)] : 0;
  return { days: xs.length, picks, avgRet: picks ? raw / picks : 0, excess: s.mean, t: s.t, logExcess: l.mean, tLog: l.t, medExcess, winVsPeers: each.length ? each.filter((v) => v > 0).length / each.length : 0 };
}
