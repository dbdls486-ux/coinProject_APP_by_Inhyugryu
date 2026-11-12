#!/usr/bin/env python3
"""
실시간 모의매매 장부 — 매시간 돈다.

  1. 바이낸스에서 지금 거래 중인 USDT 현물의 일봉을 받는다 (오늘 진행 중인 봉 포함).
  2. 아직 처리하지 않은 '끝난 날'마다: 시가 체결 → 장중 손절·익절 → 종가 처리(기간 만료, 평가, 새 주문).
  3. 오늘(진행 중): 대기 주문을 오늘 시가로 체결하고, 지금까지의 고가·저가로 손절·익절을 확인한다.
  4. 장부와 화면용 요약을 data/paper/ 에 쓴다.

매수 여부는 playbook.json(매주 backtest.py 가 갱신하는 기대값 표)을 읽어 정한다. 실제 주문은 내지 않는다.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
from core import MAJORS, P, STATES, TYPES, log  # noqa: E402


def main():
    coins, partial, today, failed = core.fetch_live()
    if "BTC" not in coins or len(coins) < int(os.environ.get("PAPER_MIN_COINS", "100")):
        raise SystemExit(f"시세를 충분히 받지 못함 ({len(coins)}종목)")
    log(f"바이낸스 {len(coins)}종목 (실패 {len(failed)}), 오늘 {today} 진행 중인 봉 {len(partial)}개")
    sig, mkt, dates = core.build_panel(coins)
    pb = core.load("playbook.json", {}) or {}
    gates = pb.get("gates", {})
    file = core.load("state.json")
    st = (file or {}).get("engine")
    ppx = lambda s: partial.get(s)
    sig_by_t = {int(t): g.to_dict("records") for t, g in sig.groupby("t")}
    if st is None:
        st = core.new_state(P["start_cash"], today)
        st["closeDone"], st["openDone"] = dates[-1], today
        # 처음 만들 때: 규칙상 이미 보유 구간인 메이저는 지금 가격으로 편입한다 (알트는 다음 신호부터)
        st["pending"] = [dict(kind="major_buy", sym=m, signalDate=dates[-1]) for m in MAJORS if m in coins and m in partial and core.major_long_now(coins[m])]
        core.step_open(st, today, lambda s: (partial[s][3],) * 4 if s in partial else None)
        st["bench"] = dict(btc0=float(partial["BTC"][3]) if "BTC" in partial else float(coins["BTC"]["c"][-1]))
        st["market"] = dict(date=dates[-1], state=str(mkt.state.iloc[-1]), btc_rsi=round(float(mkt.btc_rsi.iloc[-1]), 1), btc_d50=round(float(mkt.btc_d50.iloc[-1]), 4), mkt_brk=round(float(mkt.mkt_brk.iloc[-1]), 4), names=int(mkt.names.iloc[-1]))
        log(f"장부를 새로 만듦: 시작 자금 {P['start_cash']:,.0f} USDT, 메이저 편입 {sorted(st['majors'])}")
    else:
        for t, D in enumerate(dates):
            if D <= st["closeDone"]:
                continue
            px, ended, major_sig, stop_ref = core.make_access(coins, t)
            if st["openDone"] is None or st["openDone"] < D:
                core.step_open(st, D, px)
            core.step_intraday(st, D, px)
            core.step_close(st, D, px, ended, sig_by_t.get(t, []), mkt.iloc[t], gates, major_sig, stop_ref)
            log(f"{D} 마감 처리: 평가액 {st['lastEquity']:,.0f} · 신호 {len(st['decisions'])}건 · 새 주문 {len(st['pending'])}건")
        if partial and (st["openDone"] is None or st["openDone"] < today):
            core.step_open(st, today, ppx)
        if partial:
            core.step_intraday(st, today, ppx)

    # 지금 가격으로 평가 (하루 기록은 마감 때만 남긴다)
    now = lambda s, fallback: float(partial[s][3]) if s in partial else fallback
    for p in st["positions"]:
        p["last"] = now(p["sym"], p["last"])
    for m in st["majors"].values():
        m["last"] = now(m["sym"], m["last"])
    alt_val = sum(p["qty"] * p["last"] for p in st["positions"])
    maj_val = sum(m["qty"] * m["last"] for m in st["majors"].values())
    equity = st["cash"] + alt_val + maj_val
    st["peak"] = max(st.get("peak", equity), equity)
    hist = [e[1] for e in st["equity"]] + [equity]
    run_peak = np.maximum.accumulate(np.r_[st["initialCash"], hist])
    mdd = float((np.r_[st["initialCash"], hist] / run_peak - 1).min())
    btc_now = now("BTC", float(coins["BTC"]["c"][-1]))
    tr = st["trades"]
    alt_tr = [x for x in tr if x["sleeve"] == "alt"]
    pos_view = lambda p: dict(
        sym=p["sym"], sleeve=p["sleeve"], type=p.get("type"), typeLabel=TYPES.get(p.get("type"), "메이저 추세"), state=p.get("state"), stateLabel=STATES.get(p.get("state")), grade=p.get("grade"), entryDate=p["entryDate"], entry=p["entry"], last=p["last"],
        value=round(p["qty"] * p["last"], 2), pnlPct=p["last"] / p["entry"] - 1, stop=p.get("stop"), target=p.get("target"), risk=p.get("risk"), days=int((pd.Timestamp(today) - pd.Timestamp(p["entryDate"])).days) + 1, cell=p.get("cell"),
    )
    view = dict(
        updatedAt=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"), today=today, startedAt=st["startedAt"], initialCash=st["initialCash"], equity=round(equity, 2), cash=round(st["cash"], 2), returnPct=equity / st["initialCash"] - 1, mdd=mdd,
        exposure=(alt_val + maj_val) / equity if equity > 0 else 0, btcReturnPct=btc_now / st["bench"]["btc0"] - 1 if st.get("bench") else None, playbookAsof=pb.get("asof"),
        market=dict(st["market"] or {}, stateLabel=STATES.get((st["market"] or {}).get("state")), note=core.STATE_NOTE.get((st["market"] or {}).get("state"))),
        positions=[pos_view(p) for p in st["positions"]], majors=[pos_view(m) for m in st["majors"].values()],
        pending=[dict(kind=o["kind"], sym=o["sym"], type=o.get("type"), typeLabel=TYPES.get(o.get("type"), "메이저 추세"), grade=o.get("grade"), signalDate=o["signalDate"]) for o in st["pending"]],
        decisions=[dict(d, typeLabel=TYPES.get(d["type"]), stateLabel=STATES.get(d["state"])) for d in st.get("decisions", [])],
        stats=dict(trades=len(tr), altTrades=len(alt_tr), altWin=float(np.mean([x["ret"] > 0 for x in alt_tr])) if alt_tr else None, altAvgR=float(np.mean([x["R"] for x in alt_tr])) if alt_tr else None, realized=float(sum(x["pnl"] for x in tr))),
        lastTrades=tr[-20:][::-1], failedSymbols=failed[:20],
    )
    core.save("state.json", dict(view, engine=st))
    core.save("equity.json", dict(initialCash=st["initialCash"], btc0=(st.get("bench") or {}).get("btc0"), points=[[e[0], e[1]] for e in st["equity"]], live=[today, round(equity, 2)]))
    core.save("trades.json", tr)
    from paper_report import write_report

    write_report()
    log(f"평가액 {equity:,.0f} USDT ({(equity / st['initialCash'] - 1) * 100:+.2f}%) · 알트 {len(st['positions'])}종목 · 메이저 {sorted(st['majors'])} · 시장 상태 {view['market'].get('state')}")


if __name__ == "__main__":
    main()
