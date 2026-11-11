#!/usr/bin/env python3
"""
모의매매 규칙의 과거 재현 + 기대값 표 갱신 (매주, 그리고 lab/paper 코드가 바뀔 때).

  - 전체 기간 일봉(상장폐지 종목 포함)으로 신호와 가상 거래를 만들고,
  - 2020년부터 하루씩 장부를 굴린다. 기대값 표는 매주 월요일에 '그 전에 끝난 가상 거래'만으로 다시 계산한다.
  - 마지막 날 기준 기대값 표를 playbook.json 으로 남긴다. 실시간 장부(live.py)가 이 표를 읽어 매수 여부를 정한다.

주의: 신호 종류와 과열 기준 자체는 이 과거 자료를 보고 정한 것이다. 그래서 과거 재현 성적은 실제보다 좋게 나온다.
      진짜 검증은 실시간 장부가 쌓이는 앞으로의 기록이다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
from core import MAJORS, P, log  # noqa: E402

START = os.environ.get("PAPER_BT_START", "2020-01-01")


def curve_stats(eq, dates):
    eq = np.asarray(eq, float)
    r = np.diff(eq) / eq[:-1]
    days = max((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days, 1)
    sd = r.std()
    return dict(total=float(eq[-1] / eq[0] - 1), cagr=float((eq[-1] / eq[0]) ** (365 / days) - 1), mdd=float((eq / np.maximum.accumulate(eq) - 1).min()), sharpe=float(r.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0, days=int(days))


def by_year(eq, dates):
    s = pd.Series(np.asarray(eq, float), index=pd.to_datetime(dates))
    out = {}
    prev = None
    for y, g in s.groupby(s.index.year):
        base = prev if prev is not None else g.iloc[0]
        out[str(y)] = dict(ret=float(g.iloc[-1] / base - 1), mdd=float((g / g.cummax() - 1).min()))
        prev = g.iloc[-1]
    return out


def run(coins, sig, mkt, dates, sh):
    st = core.new_state(P["start_cash"], START)
    sig_by_t = {int(t): g.to_dict("records") for t, g in sig.groupby("t")}
    done = sh[sh.done.astype(bool)]
    t0 = next(i for i, d in enumerate(dates) if d >= START)
    gates, ok_weeks, weeks = {}, {}, 0
    state_days = {}
    for t in range(t0, len(dates)):
        date = dates[t]
        if not gates or pd.Timestamp(date).weekday() == 0:
            gates = core.compute_gates(done, date)
            weeks += 1
            for k, g in gates.items():
                if g.get("ok"):
                    d = ok_weeks.setdefault(k, dict(weeks=0, first=date, last=date))
                    d["weeks"] += 1
                    d["last"] = date
        px, ended, major_sig, stop_ref = core.make_access(coins, t)
        core.step_open(st, date, px)
        core.step_intraday(st, date, px)
        core.step_close(st, date, px, ended, sig_by_t.get(t, []), mkt.iloc[t], gates, major_sig, stop_ref)
        state_days[mkt.state.iloc[t]] = state_days.get(mkt.state.iloc[t], 0) + 1
    return st, gates, ok_weeks, weeks, state_days, t0


def main():
    t_start = time.time()
    coins = core.load_cache("binance")
    log(f"바이낸스 {len(coins)}종목 읽음")
    sig, mkt, dates = core.build_panel(coins)
    log(f"신호 {len(sig):,}건 ({dates[0]} ~ {dates[-1]}) · 종류별 {sig.type.value_counts().to_dict()}")
    sh = core.shadow(sig, coins)
    log(f"가상 거래 {int(sh.done.sum()):,}건 완료")
    st, gates, ok_weeks, weeks, state_days, t0 = run(coins, sig, mkt, dates, sh)
    eq = [e[1] for e in st["equity"]]
    ed = [e[0] for e in st["equity"]]
    stats = dict(all=curve_stats(eq, ed), byYear=by_year(eq, ed))
    i24 = next((i for i, d in enumerate(ed) if d >= "2024-01-01"), None)
    if i24 is not None and i24 < len(ed) - 30:
        stats["since2024"] = curve_stats(eq[i24:], ed[i24:])
    # 기준선: BTC 그냥 보유, 메이저 5종 그냥 보유(매일 같은 비중)
    bench = {}
    b = coins["BTC"]
    bs = pd.Series(b["c"], index=b["dates"]).reindex(ed).ffill()
    bench["btc"] = dict(all=curve_stats(bs.to_numpy(), ed), byYear=by_year(bs.to_numpy(), ed))
    rets = pd.concat([pd.Series(coins[m]["c"], index=coins[m]["dates"]).pct_change().reindex(ed) for m in MAJORS if m in coins], axis=1).mean(axis=1).fillna(0)
    mj = (1 + rets).cumprod().to_numpy()
    bench["majors"] = dict(all=curve_stats(mj, ed), byYear=by_year(mj, ed))
    if i24 is not None and i24 < len(ed) - 30:
        bench["btc"]["since2024"] = curve_stats(bs.to_numpy()[i24:], ed[i24:])
        bench["majors"]["since2024"] = curve_stats(mj[i24:], ed[i24:])
    tr = pd.DataFrame(st["trades"])
    groups = {}
    if len(tr):
        tr["year"] = tr.exitDate.str[:4]
        agg = lambda g: dict(n=int(len(g)), win=float((g.ret > 0).mean()), ret=float(g.ret.mean()), R=float(g.R.mean()) if g.R.notna().any() else None, pnl=float(g.pnl.sum()))
        groups["sleeve"] = {k: agg(g) for k, g in tr.groupby("sleeve")}
        alt = tr[tr.sleeve == "alt"]
        groups["type"] = {k: agg(g) for k, g in alt.groupby("type")}
        groups["state"] = {k: agg(g) for k, g in alt.groupby("state")}
        groups["typeState"] = {f"{k[0]}|{k[1]}|{k[2]}": agg(g) for k, g in alt.groupby(["type", "state", "flagged"])}
        groups["reason"] = {k: agg(g) for k, g in alt.groupby("reason")}
        groups["altByYear"] = {k: agg(g) for k, g in alt.groupby("year")}
        groups["majorByYear"] = {k: agg(g) for k, g in tr[tr.sleeve == "major"].groupby("year")}
    expo = float(np.mean([e[5] for e in st["equity"]]))
    wk = list(range(0, len(ed), 7))
    if wk[-1] != len(ed) - 1:
        wk.append(len(ed) - 1)
    core.save("backtest.json", dict(
        generatedAt=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M"), start=ed[0], end=ed[-1], params=P, stats=stats, bench=bench, groups=groups, avgExposure=expo, stateDays=state_days,
        gateWeeks=dict(total=weeks, cells=ok_weeks), signals={k: int(v) for k, v in sig.type.value_counts().items()},
        equity=[[ed[i], round(eq[i], 2), round(float(bs.iloc[i] / bs.iloc[0] * P["start_cash"]), 2), round(float(mj[i] * P["start_cash"]), 2)] for i in wk],
        lastTrades=st["trades"][-15:], openAtEnd=dict(alts=len(st["positions"]), majors=sorted(st["majors"])),
    ))
    final = core.compute_gates(sh[sh.done.astype(bool)], (pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    core.save("playbook.json", dict(asof=dates[-1], generatedAt=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M"), gates=final, types=core.TYPES, states=core.STATES, stateNotes=core.STATE_NOTE, params=P))
    from paper_report import write_report

    write_report()
    a = stats["all"]
    log(f"과거 재현 {ed[0]} ~ {ed[-1]}: 누적 {a['total'] * 100:+.0f}% · 연 {a['cagr'] * 100:+.1f}% · 최대 낙폭 {a['mdd'] * 100:.0f}% · 거래 {len(tr)}건 · 평균 투자 비중 {expo * 100:.0f}%")
    log(f"살 수 있는 칸 {sum(1 for g in final.values() if g.get('ok'))}개 / {len(final)}개 · 끝 ({time.time() - t_start:.0f}초)")


if __name__ == "__main__":
    main()
