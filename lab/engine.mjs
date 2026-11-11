/**
 * 백테스트 엔진 — 하루 단위, 현물 매수만 (공매도 없음).
 *
 * 하루의 순서 (미래 정보를 쓰지 않도록):
 *   1) 시가: 어제 종가에 "기간 만료"로 정한 종목 매도
 *   2) 시가: 어제 종가에 고른 매수 후보 체결 (슬리피지·수수료 반영)
 *   3) 장중: 저가가 손절가에 닿으면 손절가(갭 하락이면 시가)에 매도
 *   4) 종가: 평가, 보유 기간 확인, 내일 매수할 후보 선정
 *
 * 자금: 슬롯 N개에 평가금액의 1/N 씩. 후보가 슬롯보다 많으면 점수 높은 순.
 */

export const DEFAULTS = {
  slots: 10,
  fee: 0.0025, // 빗썸 원화마켓 기본 수수료 (한쪽). 쿠폰 적용 시 0.04%
  slipLiquid: 0.0015, // 20일 평균 거래대금 100억 이상
  slipThin: 0.003, // 그 외
  liquidValue: 1e10,
};

/**
 * @param U loadUniverse() 결과 (+ U.live[t] = [[k,i],...] 거래 가능 종목 목록)
 * @param strat STRATEGIES 의 항목
 * @param exit EXITS 의 항목
 * @param o { from, to, regimeOn(Uint8Array: 날짜별 매수 허용 여부, regimes.mjs), ...DEFAULTS }
 */
export function simulate(U, strat, exit, o) {
  const opt = { ...DEFAULTS, ...o };
  const { from, to, slots, fee } = opt;
  let cash = 1;
  const pos = [];
  let pending = [];
  const busy = new Set(); // 보유 중이거나 매수 대기 중인 종목
  const trades = [];
  const eq = new Float64Array(to - from + 1);
  const inv = new Float64Array(to - from + 1);

  const sell = (q, t, rawPx, reason) => {
    // 손절은 급락 중에 체결되므로 슬리피지를 두 배로 본다
    const exitPx = rawPx * (1 - q.slip * (reason === "손절" ? 2 : 1));
    const proceeds = q.qty * exitPx * (1 - fee);
    cash += proceeds;
    trades.push({ sym: q.k.sym, signalT: q.signalT, entryT: q.entryT, exitT: t, ret: proceeds / q.alloc - 1, reason });
    busy.delete(q.k);
  };

  for (let t = from; t <= to; t++) {
    // 1) 기간 만료 매도 (시가)
    for (let j = pos.length - 1; j >= 0; j--) {
      const q = pos[j];
      const i = q.k.idx[t];
      if (q.exitNext && i >= 0) {
        sell(q, t, q.k.o[i], "기간");
        pos.splice(j, 1);
      }
    }
    // 2) 매수 체결 (시가)
    for (const p of pending) {
      const i = p.k.idx[t];
      const alloc = Math.min(p.alloc, cash);
      if (i < 0 || alloc <= 1e-9) {
        busy.delete(p.k);
        continue;
      }
      const px = p.k.o[i] * (1 + p.slip);
      cash -= alloc;
      pos.push({ k: p.k, qty: (alloc * (1 - fee)) / px, alloc, entryT: t, signalT: p.signalT, stop: px - exit.stopAtr * p.atr, slip: p.slip, exitNext: false, last: p.k.o[i] });
    }
    pending = [];
    // 3) 손절 (장중) + 4) 종가 평가·보유 기간
    for (let j = pos.length - 1; j >= 0; j--) {
      const q = pos[j];
      const k = q.k;
      const i = k.idx[t];
      if (i < 0) continue; // 그날 봉이 없으면 다음 봉까지 그대로
      if (k.l[i] <= q.stop) {
        sell(q, t, Math.min(k.o[i], q.stop), "손절");
        pos.splice(j, 1);
        continue;
      }
      q.last = k.c[i];
      if (t - q.entryT + 1 >= exit.maxHold) q.exitNext = true;
    }
    let mv = 0;
    for (const q of pos) mv += q.qty * q.last;
    const equity = cash + mv;
    eq[t - from] = equity;
    inv[t - from] = equity > 0 ? mv / equity : 0;

    // 4) 내일 시가에 살 후보
    if (t === to) break;
    if (opt.regimeOn && !opt.regimeOn[t]) continue; // 국면 필터가 꺼진 날은 새로 사지 않는다
    // maxPicks: 무작위 대조군이 원래 규칙과 같은 날 같은 개수만 사도록 (종목 선택력만 비교)
    const free = Math.min(slots - pos.filter((q) => !q.exitNext).length, opt.maxPicks ? opt.maxPicks[t] : Infinity);
    if (free <= 0) continue;
    const cands = [];
    for (const [k, i] of U.live[t]) {
      if (busy.has(k)) continue;
      const a = k.f.atr14[i];
      if (!(a > 0)) continue;
      const s = strat.entry(k, i, t);
      if (s === null || !Number.isFinite(s)) continue;
      cands.push([s, k, i]);
    }
    if (cands.length === 0) continue;
    cands.sort((x, y) => y[0] - x[0]);
    for (const [, k, i] of cands.slice(0, free)) {
      pending.push({ k, signalT: t, alloc: equity / slots, atr: k.f.atr14[i], slip: k.f.val20[i] >= opt.liquidValue ? opt.slipLiquid : opt.slipThin });
      busy.add(k);
    }
  }
  // 마지막 날 남은 보유분은 종가로 평가만 하고 "진행 중"으로 기록
  for (const q of pos) {
    const exitPx = q.last * (1 - q.slip);
    trades.push({ sym: q.k.sym, signalT: q.signalT, entryT: q.entryT, exitT: to, ret: (q.qty * exitPx * (1 - fee)) / q.alloc - 1, reason: "진행 중" });
  }
  return { from, to, eq, inv, trades };
}

/** 기간 [a, b] (날짜 번호) 의 성적. 매매는 매수일 기준으로 그 기간에 넣는다 */
export function periodStats(run, a, b) {
  const { from, eq, inv, trades } = run;
  const at = (t) => (t < from ? 1 : eq[t - from]);
  const base = at(a - 1);
  const days = b - a + 1;
  const total = at(b) / base - 1;
  let peak = base;
  let mdd = 0;
  const rs = [];
  let prev = base;
  let expo = 0;
  for (let t = a; t <= b; t++) {
    const e = at(t);
    peak = Math.max(peak, e);
    mdd = Math.min(mdd, e / peak - 1);
    rs.push(e / prev - 1);
    prev = e;
    expo += t >= from ? inv[t - from] : 0;
  }
  const mean = rs.reduce((s, x) => s + x, 0) / rs.length;
  const sd = Math.sqrt(rs.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(1, rs.length - 1));
  const tr = trades.filter((x) => x.entryT >= a && x.entryT <= b).map((x) => x.ret);
  const n = tr.length;
  const tMean = n ? tr.reduce((s, x) => s + x, 0) / n : 0;
  const tSd = n > 1 ? Math.sqrt(tr.reduce((s, x) => s + (x - tMean) ** 2, 0) / (n - 1)) : 0;
  const wins = tr.filter((x) => x > 0);
  const losses = tr.filter((x) => x <= 0);
  const avg = (xs) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);
  return {
    days,
    total,
    cagr: total > -1 ? (1 + total) ** (365 / days) - 1 : -1,
    mdd,
    sharpe: sd > 0 ? (mean / sd) * Math.sqrt(365) : 0,
    exposure: expo / days,
    trades: n,
    winRate: n ? wins.length / n : 0,
    avgTrade: tMean,
    tStat: tSd > 0 ? tMean / (tSd / Math.sqrt(n)) : 0,
    payoff: losses.length && avg(losses) !== 0 ? avg(wins) / Math.abs(avg(losses)) : 0,
  };
}

/** 달력 연도별 수익률 */
export function yearly(run, dates) {
  const out = {};
  const { from, to, eq } = run;
  let startEq = 1;
  let year = dates[from].slice(0, 4);
  for (let t = from; t <= to; t++) {
    const y = dates[t].slice(0, 4);
    if (y !== year) {
      out[year] = eq[t - 1 - from] / startEq - 1;
      startEq = eq[t - 1 - from];
      year = y;
    }
  }
  out[year] = eq[to - from] / startEq - 1;
  return out;
}

/** 같은 기간 BTC 를 그냥 들고 있었을 때 */
export function holdStats(U, a, b) {
  const k = U.btc;
  const px = (t) => {
    for (let s = t; s >= 0; s--) if (k.idx[s] >= 0) return k.c[k.idx[s]];
    return NaN;
  };
  const base = px(a - 1 >= 0 ? a - 1 : a);
  let peak = base;
  let mdd = 0;
  const rs = [];
  let prev = base;
  for (let t = a; t <= b; t++) {
    const p = px(t);
    peak = Math.max(peak, p);
    mdd = Math.min(mdd, p / peak - 1);
    rs.push(p / prev - 1);
    prev = p;
  }
  const total = px(b) / base - 1;
  const days = b - a + 1;
  const mean = rs.reduce((s, x) => s + x, 0) / rs.length;
  const sd = Math.sqrt(rs.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(1, rs.length - 1));
  return { days, total, cagr: (1 + total) ** (365 / days) - 1, mdd, sharpe: sd > 0 ? (mean / sd) * Math.sqrt(365) : 0 };
}
