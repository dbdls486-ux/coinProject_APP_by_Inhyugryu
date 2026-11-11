"""indicators.json → INDICATORS.md"""
from features import BINARY


def pc(x, d=1):
    return "-" if x is None else f"{x * 100:+.{d}f}%"


def pp(x, d=2):
    return "-" if x is None else f"{x * 100:+.{d}f}%p"


def num(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def render(R):
    L = []
    A = L.append
    A("# 급등 예측 지표 검증")
    A("")
    A(f"생성 {R['generatedAt'].replace('T', ' ')} UTC · 오늘 종가에 고르고 내일 시가에 사서 {R['horizon']}일 보유 · 하루 {R['picksPerDay']}종목")
    A("")
    A("**급등** = 매수 후 7일 안에 매수가 대비 +30% 이상 오른 적이 있음. **실수익** = 7일 뒤 종가에 팔았을 때 수수료·슬리피지를 뺀 수익. **초과** = 같은 날 비슷하게 출렁이는 코인 평균 대비.")
    A("")
    A("## 한눈에")
    for m in R["markets"]:
        sc = {s["id"]: s for s in m["scores"]}
        A(f"### {m['label']} ({m['coins']}종목, {m['dates'][0]} ~ {m['dates'][1]})")
        A(f"- 아무 코인이나 샀을 때 급등 확률: 고르는 기간 {pc(m['base']['train'])}, 확인 기간 {pc(m['base']['test'])}")
        real = [r for r in m["scan"] if not r["placebo"]]
        fake = [r for r in m["scan"] if r["placebo"]]
        fake_max = max(abs(r["train"]["ic_t"] or 0) for r in fake)
        both = [r for r in real if r["stable"] and abs(r["train"]["ic_t"] or 0) >= 3 and abs(r["test"]["ic_t"] or 0) >= 3 and (r["train"]["ic"] > 0) == (r["test"]["ic"] > 0)]
        A(f"- 지표 {len(real)}개 중 두 기간 모두 뚜렷하고(|t|≥3) 방향이 같은 것: {len(both)}개 (가짜 지표의 최대 |t|는 {num(fake_max)})")
        for sid in ("s_rand@oos", "s_vol@oos", "s_comp@oos", "s_ml_pump@oos", "s_ml_ret@oos", "s_ml_both@oos"):
            s = sc.get(sid)
            if not s or "test" not in s["periods"]:
                continue
            p = s["periods"]["test"]
            A(f"- {s['label']}: 확인 기간 급등 적중 {pc(p['pump'])} (기준 {pc(p['base'])}), 거래당 실수익 {pc(p['net'])} (중앙값 {pc(p['median'])}), 계좌 연 {pc(p['cagr'])} / 최대 낙폭 {pc(p['mdd'])}")
        A(f"- BTC 그냥 보유: 확인 기간 연 {pc(m['btcHold']['test']['cagr'])} / 최대 낙폭 {pc(m['btcHold']['test']['mdd'])}")
        A("")

    for m in R["markets"]:
        A(f"## {m['label']}")
        A("")
        A("### 1. 섞은 점수로 매일 상위 5종목을 샀다면")
        A("기계학습 모델은 매년 1월 1일 이전 자료로만 학습해 그 해에 적용했다. 아래 표는 모델 예측이 있는 구간(2021년~)에서 모든 방식을 같은 날짜로 비교한 것이다.")
        A("")
        A("| 방식 | 기간 | 거래 | 급등 적중 (기준) | 거래당 실수익 | 중앙값 | 이긴 비율 | 초과 (t) | 계좌 연수익 | 최대 낙폭 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for s in m["scores"]:
            if not s["id"].endswith("@oos"):
                continue
            for key, name in (("train", "2021~2023"), ("test", "2024~")):
                p = s["periods"].get(key)
                if p:
                    A(f"| {s['label']} | {name} | {p['trades']} | {pc(p['pump'])} ({pc(p['base'])}) | {pc(p['net'])} | {pc(p['median'])} | {round(p['win'] * 100)}% | {pp(p['ex'])} ({num(p['t'])}) | {pc(p['cagr'])} | {pc(p['mdd'])} |")
        A("")
        years = sorted({y for s in m["scores"] if s["id"].endswith("@oos") for y in s["years"]})
        A("**연도별 계좌 수익**")
        A("")
        A(f"| 방식 | {' | '.join(years)} |")
        A(f"|---|{'|'.join('---' for _ in years)}|")
        for s in m["scores"]:
            if s["id"].endswith("@oos"):
                A(f"| {s['label']} | {' | '.join(pc(s['years'].get(y), 0) for y in years)} |")
        A("")
        A("**익절·손절을 붙이면** (확인 기간, 거래당 평균 / 중앙값 / 이긴 비율)")
        A("")
        A("| 방식 | 7일 보유 | +30% 익절·−10% 손절 | +50% 익절·−15% 손절 | +20% 익절·−7% 손절 | −10% 손절만 |")
        A("|---|---|---|---|---|---|")
        for s in m["scores"]:
            p = s["periods"].get("test")
            if not s["id"].endswith("@oos") or not p:
                continue
            e = p["exits"]
            cell = lambda x: f"{pc(x['mean'])} / {pc(x['median'])} / {round(x['win'] * 100)}%"
            A(f"| {s['label']} | {pc(p['net'])} / {pc(p['median'])} / {round(p['win'] * 100)}% | {cell(e['tp30sl10'])} | {cell(e['tp50sl15'])} | {cell(e['tp20sl7'])} | {cell(e['sl10'])} |")
        A("")
        if m["composite"]:
            A("**합성 점수에 들어간 지표** (고르는 기간 기준): " + ", ".join(f"{c['label']}({'높을수록' if c['sign'] > 0 else '낮을수록'})" for c in m["composite"]))
            A("")
        if m["importance"]["pump"]:
            A("**급등 확률 모델이 많이 기댄 지표** (섞었을 때 적중률이 가장 많이 떨어지는 순): " + ", ".join(f"{LABELS(k)}" for k, _ in m["importance"]["pump"][:10]))
            A("")
            A("**초과수익 모델이 많이 기댄 지표**: " + ", ".join(f"{LABELS(k)}" for k, _ in m["importance"]["ret"][:10]))
            A("")

        A("### 2. 개별 지표 채점표")
        A("IC = 그날 지표 순위와 이후 7일 초과수익 순위의 상관 (양수면 지표가 높을수록 이후 수익이 좋았다). 급등 배수 = 지표 상위 10% 코인의 급등 확률 ÷ 전체 급등 확률.")
        A("")
        A("| 지표 | IC 고르는 기간 (t) | IC 확인 기간 (t) | 상위 10% 급등 배수 (고름/확인) | 하위 10% 급등 배수 | 상위 10% 실수익 (고름/확인) | 하위 10% 실수익 | 상위 10% 초과 확인 기간 (t) |")
        A("|---|---|---|---|---|---|---|---|")
        rows = sorted(m["scan"], key=lambda r: -min(abs(r["train"]["ic_t"] or 0), abs(r["test"]["ic_t"] or 0)) if (r["train"]["ic"] or 0) * (r["test"]["ic"] or 0) > 0 else 0)
        for r in rows:
            a, b = r["train"], r["test"]
            head = f"| {'*' if r['placebo'] else ''}{r['label']} | {num(a['ic'] * 100 if a['ic'] is not None else None, 2)} ({num(a['ic_t'])}) | {num(b['ic'] * 100 if b['ic'] is not None else None, 2)} ({num(b['ic_t'])}) |"
            if r["feature"] in BINARY:  # 켜짐/꺼짐뿐인 신호는 상·하위 10% 를 나눌 수 없다 → 3번 표 참고
                A(f"{head} 3번 표 | - | - | - | - |")
                continue
            A(f"{head} {num(a['top_lift'], 2)} / {num(b['top_lift'], 2)} | {num(a['bot_lift'], 2)} / {num(b['bot_lift'], 2)} | {pc(a['top_net'])} / {pc(b['top_net'])} | {pc(a['bot_net'])} / {pc(b['bot_net'])} | {pp(b['top_ex'])} ({num(b['top_ex_t'])}) |")
        A("")
        A("\\* 표시는 무작위 가짜 지표. IC 값은 ×100.")
        A("")

        A("### 3. 신호형 지표 (신호가 뜬 날 다음 날 매수)")
        A("| 신호 | 건수 (고름/확인) | 급등 배수 | 초과 고르는 기간 (t) | 초과 확인 기간 (t) | 실수익 평균 (고름/확인) | 확인 기간 중앙값 | 이긴 비율 |")
        A("|---|---|---|---|---|---|---|---|")
        sig = sorted(m["signals"], key=lambda s: -(s["test"].get("ex") or -9))
        for s in sig:
            a, b = s["train"], s["test"]
            if a.get("n", 0) < 30 or b.get("n", 0) < 30:
                continue
            A(f"| {s['label']} | {a['n']} / {b['n']} | {num(a['lift'], 2)} / {num(b['lift'], 2)} | {pp(a['ex'])} ({num(a['t'])}) | {pp(b['ex'])} ({num(b['t'])}) | {pc(a['net'])} / {pc(b['net'])} | {pc(b['median'])} | {round(b['win'] * 100)}% |")
        A("")

        A("### 4. 신호 2개 조합 — 두 기간 모두 초과수익이 좋았던 상위 15개와 나빴던 상위 10개")
        best = m["pairs"]["best"][:15]
        worst = m["pairs"]["worst"][:10]
        A(f"조합 {m['pairCount']}개를 봤다. 이 정도 개수면 우연히 좋아 보이는 조합이 여러 개 나오므로 t가 3을 넘는지와 실수익을 같이 본다.")
        A("")
        A("| 조합 | 건수 (고름/확인) | 급등 배수 | 초과 고르는 기간 (t) | 초과 확인 기간 (t) | 실수익 평균 (고름/확인) |")
        A("|---|---|---|---|---|---|")
        for p in best + worst:
            a, b = p["train"], p["test"]
            A(f"| {p['label']} | {a['n']} / {b['n']} | {num(a['lift'], 2)} / {num(b['lift'], 2)} | {pp(a['ex'])} ({num(a['t'])}) | {pp(b['ex'])} ({num(b['t'])}) | {pc(a['net'])} / {pc(b['net'])} |")
        A("")

        for sym in ("BTC", "ETH"):
            rows = m["single"].get(sym) or []
            if not rows:
                continue
            A(f"### 5. {sym} 한 종목에 신호를 적용하면 (신호가 뜬 다음 날 매수, 평소 대비)")
            A("| 신호 | 건수 (고름/확인) | 7일 뒤 고르는 기간 (평소) | 7일 뒤 확인 기간 (평소) | 28일 뒤 고르는 기간 (평소) | 28일 뒤 확인 기간 (평소) |")
            A("|---|---|---|---|---|---|")
            for r in rows:
                a, b = r["train"], r["test"]
                if a["n"] < 10 or b["n"] < 5:
                    continue
                A(f"| {r['label']} | {a['n']} / {b['n']} | {pc(a['f7'])} ({pc(a['base7'])}) | {pc(b['f7'])} ({pc(b['base7'])}) | {pc(a['f28'])} ({pc(a['base28'])}) | {pc(b['f28'])} ({pc(b['base28'])}) |")
            A("")

        if m.get("round2"):
            render_round2(A, m["round2"])

    A("## 읽을 때 주의")
    A("- 2차 실험(6번)은 조건을 여러 개 바꿔 본 것이라, 그중 하나가 좋아 보여도 고르는 기간과 확인 기간이 같이 좋은지를 봐야 한다.")
    A("- 급등 적중률이 높아도 실수익이 마이너스일 수 있다. 급등하는 코인은 급락도 잦아서, 급등 확률이 높은 코인만 고르면 평균 수익은 오히려 나빠지기 쉽다.")
    A("- 급등은 장중 고가 기준이다. 실제로 그 가격에 팔 수 있었다는 뜻이 아니다. 익절·손절 표는 고가·저가가 그 가격에 닿으면 체결됐다고 가정했고, 같은 날 둘 다 닿으면 손절이 먼저라고 봤다.")
    A("- 빗썸은 지금 상장된 종목만 들어 있다 (상장폐지 종목 빠짐). 바이낸스는 상장폐지 종목을 포함한다.")
    A("- 지표 100개, 조합 수백 개를 한꺼번에 보면 우연히 좋아 보이는 것이 반드시 나온다. 가짜 지표의 값과 두 기간의 일치 여부를 기준으로 삼는다.")
    return "\n".join(L) + "\n"


def LABELS(k):
    from features import LABEL

    extra = {"btc_d_sma50": "BTC 50일선 대비 위치", "btc_roc7": "BTC 7일 등락률", "btc_roc28": "BTC 28일 등락률", "btc_rsi14": "BTC RSI", "breadth50": "50일선 위 코인 비율", "mkt_ret1": "시장 중앙값 1일 등락", "mkt_ret7": "시장 중앙값 7일 등락"}
    return LABEL.get(k) or extra.get(k) or k


def _cell(c):
    if not c or "net" not in c:
        return "-"
    return f"{pc(c['net'])} / {pc(c['median'])} / {pp(c['ex'])} ({num(c['t'])})"


def _acct(a):
    return "-" if not a else f"{pc(a['cagr'], 0)} / {pc(a['mdd'], 0)}"


def render_round2(A, r2):
    A("### 6. 2차 실험 — 조건을 바꿔 가며 다시 확인")
    A("")
    if r2.get("regimes"):
        A("#### 6-A. 상승장일 때만 사면")
        A("칸 안의 숫자는 거래당 실수익 평균 / 중앙값 / 초과 (t). 계좌는 조건이 켜진 날에만 새로 사고 나머지는 현금으로 둔 경우의 연수익 / 최대 낙폭.")
        A("")
        for g in r2["regimes"]:
            A(f"**{g['label']}** — 켜진 날 비율: 2021~2023년 {round(g['share']['train'] * 100)}%, 2024년~ {round(g['share']['test'] * 100)}%")
            A("")
            A("| 방식 | 2024~ 켜진 날 급등 적중 (기준) | 2024~ 켜진 날 | 2024~ 꺼진 날 | 2021~2023 켜진 날 | 2021~2023 꺼진 날 | 켜진 날만 산 계좌 2024~ | 2021~2023 |")
            A("|---|---|---|---|---|---|---|---|")
            for r in g["rows"]:
                c = r["cells"]
                on = c.get("test_on") or {}
                hit = f"{pc(on.get('pump'))} ({pc(on.get('base'))})" if "pump" in on else "-"
                A(f"| {r['label']} | {hit} | {_cell(c.get('test_on'))} | {_cell(c.get('test_off'))} | {_cell(c.get('train_on'))} | {_cell(c.get('train_off'))} | {_acct(r['account'].get('test'))} | {_acct(r['account'].get('train'))} |")
            A("")

    cal = r2.get("calibration")
    if cal:
        A("#### 6-B. 모델이 확신할 때만 사면")
        A("모델이 낸 확률을 구간으로 나눠, 그 구간에 든 종목을 전부 샀다고 봤다.")
        A("")
        for key, title, hit in (("pump", "급등 확률 모델", "실제 급등 비율"), ("ret", "초과수익 모델", "실제로 상위 20%에 든 비율")):
            A(f"**{title}**")
            A("")
            A(f"| 예측 확률 구간 | 기간 | 건수 (날 수) | {hit} | 실수익 평균 | 중앙값 | 이긴 비율 | 초과 (t) | +30% 익절·−10% 손절 |")
            A("|---|---|---|---|---|---|---|---|---|")
            for r in cal[key]:
                if r["n"] < 30:
                    continue
                A(f"| {round(r['lo'] * 100)}~{min(100, round(r['hi'] * 100))}% | {'2021~2023' if r['period'] == 'train' else '2024~'} | {r['n']} ({r['days']}) | {pc(r['hit'])} | {pc(r['net'])} | {pc(r['median'])} | {round(r['win'] * 100)}% | {pp(r['ex'])} ({num(r['t'])}) | {pc(r['tpsl'])} |")
            A("")

    if r2.get("ksweep"):
        A("#### 6-C. 하루에 고르는 종목 수를 바꾸면")
        A("| 방식 | 하루 종목 수 | 2024~ 급등 적중 | 실수익 평균 | 중앙값 | 초과 (t) | 2021~2023 실수익 평균 | 초과 (t) |")
        A("|---|---|---|---|---|---|---|---|")
        for r in r2["ksweep"]:
            a, b = r["train"], r["test"]
            if "net" not in a or "net" not in b:
                continue
            A(f"| {r['label']} | {r['k']} | {pc(b['pump'])} | {pc(b['net'])} | {pc(b['median'])} | {pp(b['ex'])} ({num(b['t'])}) | {pc(a['net'])} | {pp(a['ex'])} ({num(a['t'])}) |")
        A("")

    lq = r2.get("liquid")
    if lq:
        A("#### 6-G. 거래대금 상위 30% 종목 안에서만 고르면")
        A(f"하루 평균 {round(lq['names'])}종목 안에서 5종목을 고른다.")
        A("")
        A("| 방식 | 기간 | 거래 | 급등 적중 (기준) | 실수익 평균 | 중앙값 | 이긴 비율 | 초과 (t) | 계좌 연수익 / 최대 낙폭 |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in lq["rows"]:
            for key, nm in (("train", "2021~2023"), ("test", "2024~")):
                p = r.get(key) or {}
                if "net" not in p:
                    continue
                A(f"| {r['label']} | {nm} | {p['trades']} | {pc(p['pump'])} ({pc(p['base'])}) | {pc(p['net'])} | {pc(p['median'])} | {round(p['win'] * 100)}% | {pp(p['ex'])} ({num(p['t'])}) | {_acct(r['account'].get(key))} |")
        A("")

    if r2.get("exits"):
        A("#### 6-D. 익절·손절·보유 기간 조합 (3·7·14일 × 익절 5가지 × 손절 5가지)")
        A("| 방식 | 평균이 플러스인 조합 수: 2021~2023 / 2024~ / 둘 다 | 2021~2023에 가장 좋았던 조합 | 그 조합의 2021~2023 평균 | 같은 조합의 2024~ 평균 (중앙값, 이긴 비율) |")
        A("|---|---|---|---|---|")
        name = lambda c: f"{c['hz']}일 보유 · 익절 {('+' + str(round(c['tp'] * 100)) + '%') if c['tp'] else '없음'} · 손절 {('−' + str(round(c['sl'] * 100)) + '%') if c['sl'] else '없음'}"
        for e in r2["exits"]:
            b = e.get("bestByTrain")
            if not b:
                continue
            A(f"| {e['label']} | {e['positiveTrain']} / {e['positiveTest']} / {e['positiveBoth']} (전체 {e['combos']}) | {name(b)} | {pc(b['train']['mean'])} | {pc(b['test']['mean'])} ({pc(b['test']['median'])}, {round(b['test']['win'] * 100)}%) |")
        A("")
        for e in r2["exits"]:
            if e["id"] not in ("s_ml_pump", "s_ml_ret"):
                continue
            for hz in sorted({c["hz"] for c in e["cells"]}):
                cells = {(c["tp"], c["sl"]): c for c in e["cells"] if c["hz"] == hz and "test" in c}
                if not cells:
                    continue
                tps = sorted({k[0] for k in cells})
                sls = sorted({k[1] for k in cells})
                A(f"**{e['label']} · {hz}일 보유 · 2024~ 거래당 평균 실수익** (세로 익절, 가로 손절)")
                A("")
                A("| 익절 \\ 손절 | " + " | ".join(("−" + str(round(x * 100)) + "%") if x else "없음" for x in sls) + " |")
                A("|---|" + "|".join("---" for _ in sls) + "|")
                for tp in tps:
                    A(f"| {('+' + str(round(tp * 100)) + '%') if tp else '없음'} | " + " | ".join(pc(cells[(tp, sl)]["test"]["mean"]) if (tp, sl) in cells else "-" for sl in sls) + " |")
                A("")

    lg = r2.get("long")
    if lg:
        A("#### 6-E. 4주(28일) 보유로 길게 보면")
        A(f"급등 = 28일 안에 +50% 이상. 아무 코인이나 샀을 때 급등 확률: 2021~2023년 {pc(lg['base']['train'])}, 2024년~ {pc(lg['base']['test'])}. 모델은 이 목표로 따로 학습했다.")
        A("")
        A("| 방식 | 기간 | 거래 | 급등 적중 | 실수익 평균 | 중앙값 | 이긴 비율 | 초과 (t) | 계좌 연수익 / 최대 낙폭 |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in lg["rows"]:
            for key, nm in (("train", "2021~2023"), ("test", "2024~")):
                p = r.get(key) or {}
                if "net" not in p:
                    continue
                A(f"| {r['label']} | {nm} | {p['trades']} | {pc(p['pump'])} | {pc(p['net'])} | {pc(p['median'])} | {round(p['win'] * 100)}% | {pp(p['ex'])} ({num(p['t'])}) | {_acct(r['account'].get(key))} |")
        A("")

    if r2.get("majors"):
        A("#### 6-F. 같은 지표를 메이저 코인에 규칙으로 쓰면 (BTC·ETH·XRP·SOL·DOGE)")
        A("판단은 종가에, 체결은 다음 날 시가에. 칸 안은 연수익 / 최대 낙폭. 바구니는 다섯 종목을 같은 비중으로 각자 규칙대로 굴린 것.")
        A("")
        syms = [s for s in ("BTC", "ETH", "XRP", "SOL", "DOGE") if any(s in r["coins"] for r in r2["majors"])]
        A("| 규칙 | 종류 | 바구니 ~2023 | 바구니 2024~ | " + " | ".join(f"{s} ~2023 | {s} 2024~" for s in syms) + " | BTC 보유일 비율 2024~ |")
        A("|---|---|---|---|" + "|".join("---|---" for _ in syms) + "|---|")
        for r in r2["majors"]:
            cells = []
            for s in syms:
                d = r["coins"].get(s, {})
                cells += [_acct(d.get("train")), _acct(d.get("test"))]
            expo = (r["coins"].get("BTC", {}).get("test") or {}).get("exposure")
            A(f"| {r['label']} | {r['kind']} | {_acct(r['basket'].get('train'))} | {_acct(r['basket'].get('test'))} | " + " | ".join(cells) + f" | {'-' if expo is None else str(round(expo * 100)) + '%'} |")
        A("")
