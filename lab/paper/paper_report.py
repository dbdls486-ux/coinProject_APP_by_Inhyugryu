"""data/paper/*.json → lab/results/PAPER.md"""
import os

import core
from core import STATES, STATE_NOTE, TYPES

OUT_MD = os.environ.get("PAPER_MD", "lab/results/PAPER.md")


def pc(x, d=1):
    return "-" if x is None else f"{x * 100:+.{d}f}%"


def num(x, d=2):
    return "-" if x is None else f"{x:+.{d}f}"


def usd(x):
    return "-" if x is None else f"{x:,.0f}"


def px(x):
    if x is None:
        return "-"
    return f"{x:,.2f}" if x >= 100 else (f"{x:.4f}" if x >= 0.1 else f"{x:.6g}")


def write_report():
    bt, pb, st = core.load("backtest.json"), core.load("playbook.json"), core.load("state.json")
    L = []
    A = L.append
    A("# 모의매매")
    A("")
    A("실제 주문은 내지 않는다. 바이낸스 USDT 현물 일봉으로 4시간마다 장부를 갱신한다.")
    A("")
    A("## 규칙")
    A("- **메이저 묶음** (BTC·ETH·XRP·SOL·DOGE, 종목당 자금의 8%): 20일 고점 돌파에 사서 10일 최저 종가를 깨면 판다.")
    A("- **알트 묶음** (최대 12종목, 합계 55%까지): 신호가 뜨면 아래 '기대값 표'에서 그 상황의 과거 성적을 찾아, 통계적으로 플러스인 칸만 산다.")
    A("  - 손절 = 최근 10일 저가의 1% 아래 (매수가의 4~25% 범위), 익절 = 위험의 2배, 최대 28일 보유.")
    A("  - 비중 = 손절까지 가면 자금의 0.75%(A등급) 또는 0.4%(B등급)를 잃도록 역산, 종목당 최대 8%.")
    A("  - 신호는 종가에 판단하고 다음 날 시가에 산다. 수수료 0.1% + 체결 오차 0.15%를 매수·매도마다 뺀다.")
    A("- **시장 상태**: " + " / ".join(f"{STATES[k]} = {STATE_NOTE[k]}" for k in STATES) + " (위에서부터 먼저 해당하는 것)")
    A("- **과열** = 신호 당일 +15% 이상, 또는 28일간 BTC보다 20%p 넘게 앞섬, 또는 30일 안에 하루 +29% 넘게 오른 날이 있었음.")
    A("- **살 수 있는 칸** = 끝난 가상 거래 50건·10주 이상 + (평균 − 오차)가 0보다 큼 + 최근 1년 평균이 마이너스가 아님. 기대값 표는 매주 다시 계산한다.")
    A("")
    if st:
        m = st.get("market") or {}
        A(f"## 지금 장부 ({st['updatedAt'].replace('T', ' ').replace('Z', ' UTC')})")
        A("")
        A(f"- 평가액 **{usd(st['equity'])} USDT** (시작 {usd(st['initialCash'])}, {st['startedAt']}) · 수익률 **{pc(st['returnPct'], 2)}** · 같은 기간 BTC {pc(st.get('btcReturnPct'), 2)} · 최대 낙폭 {pc(st['mdd'], 1)}")
        A(f"- 현금 {usd(st['cash'])} USDT · 투자 비중 {pc(st['exposure'], 0).replace('+', '')} · 끝난 거래 {st['stats']['trades']}건")
        A(f"- 시장 상태 ({m.get('date')}): **{m.get('stateLabel')}** — BTC RSI {m.get('btc_rsi')}, BTC 50일선 대비 {pc(m.get('btc_d50'))}, 거래량 동반 60일 고점 돌파 종목 {pc(m.get('mkt_brk'), 1).replace('+', '')} (대상 {m.get('names')}종목)")
        A(f"- 기대값 표 기준일: {st.get('playbookAsof') or '아직 없음'}")
        A("")
        rows = st["majors"] + st["positions"]
        if rows:
            A("| 종목 | 묶음 | 신호 | 매수일 | 매수가 | 현재가 | 손익 | 평가액 | 손절가 | 목표가 | 보유일 |")
            A("|---|---|---|---|---|---|---|---|---|---|---|")
            for p in rows:
                A(f"| {p['sym']} | {'메이저' if p['sleeve'] == 'major' else '알트 ' + (p.get('grade') or '')} | {p['typeLabel']} | {p['entryDate']} | {px(p['entry'])} | {px(p['last'])} | {pc(p['pnlPct'])} | {usd(p['value'])} | {px(p.get('stop'))} | {px(p.get('target'))} | {p['days']} |")
            A("")
        else:
            A("보유 종목 없음.")
            A("")
        if st["pending"]:
            A("**다음 시가에 체결될 주문**: " + ", ".join(f"{o['sym']} {'매도' if o['kind'].endswith('sell') else '매수'} ({o['typeLabel']})" for o in st["pending"]))
            A("")
        if st["decisions"]:
            A(f"**마지막 마감일({m.get('date')})의 알트 신호와 결정**")
            A("")
            A("| 종목 | 신호 | 시장 상태 | 과열 | 과거 표본 | 평균 R | 하한 | 결정 | 이유 |")
            A("|---|---|---|---|---|---|---|---|---|")
            for d in st["decisions"]:
                A(f"| {d['sym']} | {d['typeLabel']} | {d['stateLabel']} | {'예' if d['flagged'] else '-'} | {d.get('n', 0)} | {num(d.get('meanR'))} | {num(d.get('lb'))} | {d['decision']} | {d['why']} |")
            A("")
        if st["lastTrades"]:
            A("**최근 끝난 거래**")
            A("")
            A("| 종목 | 묶음 | 매수일 → 매도일 | 수익 | R | 손익 (USDT) | 사유 |")
            A("|---|---|---|---|---|---|---|")
            for t in st["lastTrades"][:15]:
                A(f"| {t['sym']} | {'메이저' if t['sleeve'] == 'major' else TYPES.get(t.get('type'), '알트')} | {t['entryDate']} → {t['exitDate']} | {pc(t['ret'])} | {num(t.get('R'))} | {t['pnl']:+,.0f} | {t['reason']} |")
            A("")
    if pb:
        A(f"## 기대값 표 ({pb['asof']} 기준)")
        A("R = 수익을 처음 정한 위험(손절폭)으로 나눈 값. 평균 R이 +0.2면 손절폭의 20%만큼을 거래당 평균적으로 벌었다는 뜻. 하한 = 평균 − 오차(주 단위로 묶어 계산).")
        A("")
        A("| 신호 | 시장 상태 | 과열 | 표본 (주) | 평균 R | 하한 | 평균 수익 | 이긴 비율 | 최근 1년 표본 · 평균 R | 판정 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for ty in TYPES:
            for s in STATES:
                for fl in (0, 1):
                    g = pb["gates"].get(f"{ty}|{s}|{fl}") or {}
                    if not g.get("n"):
                        continue
                    verdict = f"매수 {g['grade']}등급" if g.get("ok") else ("표본 부족" if g.get("level") == "표본 부족" else "관찰만")
                    if g.get("level") == "종류×상태":
                        verdict += " (과열 구분 없이 합친 성적 기준)"
                    A(f"| {TYPES[ty]} | {STATES[s]} | {'예' if fl else '-'} | {g['n']} ({g.get('weeks', 0)}) | {num(g.get('meanR'))} | {num(g.get('lb'))} | {pc(g.get('meanRet'))} | {pc(g.get('win'), 0).replace('+', '')} | {g.get('recentN', 0)} · {num(g.get('recentR'))} | {verdict} |")
        A("")
    if bt:
        a, b = bt["stats"], bt["bench"]
        A(f"## 과거 재현 ({bt['start']} ~ {bt['end']})")
        A("같은 규칙·같은 코드로 과거를 하루씩 다시 굴린 것. 기대값 표는 매주 그 시점까지 끝난 거래만으로 계산했다.")
        A("")
        A("| | 누적 | 연환산 | 최대 낙폭 | 샤프 |")
        A("|---|---|---|---|---|")
        for name, s in (("모의매매 규칙", a["all"]), ("BTC 그냥 보유", b["btc"]["all"]), ("메이저 5종 그냥 보유", b["majors"]["all"])):
            A(f"| {name} | {pc(s['total'], 0)} | {pc(s['cagr'])} | {pc(s['mdd'], 0)} | {s['sharpe']:.2f} |")
        if a.get("since2024"):
            for name, s in (("2024년 이후: 모의매매 규칙", a["since2024"]), ("2024년 이후: BTC 그냥 보유", b["btc"]["since2024"]), ("2024년 이후: 메이저 5종 그냥 보유", b["majors"]["since2024"])):
                A(f"| {name} | {pc(s['total'], 0)} | {pc(s['cagr'])} | {pc(s['mdd'], 0)} | {s['sharpe']:.2f} |")
        A("")
        A(f"평균 투자 비중 {bt['avgExposure'] * 100:.0f}% (나머지는 현금).")
        A("")
        years = sorted(a["byYear"])
        A("| 연도 | " + " | ".join(years) + " |")
        A("|---|" + "|".join("---" for _ in years) + "|")
        A("| 모의매매 규칙 수익 | " + " | ".join(pc(a["byYear"][y]["ret"], 0) for y in years) + " |")
        A("| 모의매매 규칙 그 해 최대 낙폭 | " + " | ".join(pc(a["byYear"][y]["mdd"], 0) for y in years) + " |")
        A("| BTC 그냥 보유 수익 | " + " | ".join(pc(b["btc"]["byYear"].get(y, {}).get("ret"), 0) for y in years) + " |")
        A("| 메이저 5종 그냥 보유 수익 | " + " | ".join(pc(b["majors"]["byYear"].get(y, {}).get("ret"), 0) for y in years) + " |")
        A("")
        g = bt.get("groups") or {}
        if g:
            row = lambda name, s: f"| {name} | {s['n']} | {pc(s['win'], 0).replace('+', '')} | {pc(s['ret'])} | {num(s.get('R'))} | {s['pnl']:+,.0f} |"
            A("| 거래 구분 | 건수 | 이긴 비율 | 평균 수익 | 평균 R | 손익 합계 (USDT, 시작 자금 10,000) |")
            A("|---|---|---|---|---|---|")
            for k, s in g.get("sleeve", {}).items():
                A(row("메이저 묶음" if k == "major" else "알트 묶음", s))
            for k, s in g.get("type", {}).items():
                A(row("알트 · " + TYPES.get(k, k), s))
            for k, s in g.get("state", {}).items():
                A(row("알트 · 시장 상태: " + STATES.get(k, k), s))
            for k, s in g.get("reason", {}).items():
                A(row("알트 · 끝난 사유: " + k, s))
            A("")
            if g.get("altByYear"):
                ys = sorted(set(g["altByYear"]) | set(g.get("majorByYear", {})))
                A("| 연도별 손익 (USDT) | " + " | ".join(ys) + " |")
                A("|---|" + "|".join("---" for _ in ys) + "|")
                A("| 알트 묶음 (건수) | " + " | ".join(f"{g['altByYear'][y]['pnl']:+,.0f} ({g['altByYear'][y]['n']})" if y in g["altByYear"] else "-" for y in ys) + " |")
                A("| 메이저 묶음 (건수) | " + " | ".join(f"{g['majorByYear'][y]['pnl']:+,.0f} ({g['majorByYear'][y]['n']})" if y in g.get("majorByYear", {}) else "-" for y in ys) + " |")
                A("")
        gw = bt.get("gateWeeks") or {}
        if gw.get("cells"):
            A(f"**칸별로 '살 수 있음'이었던 주** (전체 {gw['total']}주 중)")
            A("")
            A("| 신호 | 시장 상태 | 과열 | 살 수 있었던 주 | 처음 | 마지막 |")
            A("|---|---|---|---|---|---|")
            for k, v in sorted(gw["cells"].items(), key=lambda kv: -kv[1]["weeks"]):
                ty, s, fl = k.split("|")
                A(f"| {TYPES.get(ty, ty)} | {STATES.get(s, s)} | {'예' if fl == '1' else '-'} | {v['weeks']} | {v['first']} | {v['last']} |")
            A("")
        sd = bt.get("stateDays") or {}
        if sd:
            tot = sum(sd.values())
            A("시장 상태별 날 수: " + ", ".join(f"{STATES.get(k, k)} {v}일 ({v / tot * 100:.0f}%)" for k, v in sorted(sd.items(), key=lambda kv: -kv[1])))
            A("")
    A("## 주의")
    A("- 신호 종류와 과열 기준은 이 과거 자료를 분석해서 정한 것이다. 그래서 과거 재현 성적은 실제보다 좋게 나온다. 진짜 검증은 '지금 장부'에 앞으로 쌓이는 기록이다.")
    A("- 과거 재현에는 상장폐지된 종목도 들어 있다. 실시간 장부는 지금 거래 중인 종목만 본다.")
    A("- 체결은 '다음 날 시가'로 가정한다. 실제로는 장부가 도는 시각이 시가보다 몇 분~몇십 분 늦다.")
    A("- 장중 손절·익절은 일봉의 고가·저가로 판단한다. 과거 재현에서는 같은 날 둘 다 닿으면 손절로 쳤고, 실시간에서는 4시간마다 확인한 순서대로 처리한다.")
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
