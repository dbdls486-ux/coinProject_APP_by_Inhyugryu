"""setups.json → SETUPS.md"""


def pc(x, d=1):
    return "-" if x is None else f"{x * 100:+.{d}f}%"


def pp(x, d=2):
    return "-" if x is None else f"{x * 100:+.{d}f}%p"


def num(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def ext(p, t="t", e="ex"):
    return "-" if p.get(e) is None else f"{pp(p[e])} ({num(p.get(t))})"


def render(R):
    L = []
    A = L.append
    A("# 단계형 시나리오 · 주봉 결합 · 뉴스 결합 검증")
    A("")
    A(f"생성 {R['generatedAt'].replace('T', ' ')} UTC · 신호는 종가에, 매수는 다음 날 시가에 · 수수료·체결 오차 반영 · 고르는 기간 ~2023 / 확인 기간 2024~")
    A("")
    A("**초과** = 같은 날 비슷하게 출렁이는 코인 평균 대비. 괄호 안 t는 우연이 아닐 가능성의 척도로, 3 이상이면 뚜렷, 2 미만이면 우연과 구분이 안 된다.")
    A("**판정 '통과'** = 고르는 기간과 확인 기간 모두 초과수익의 t가 2 이상이고 확인 기간 실수익이 플러스 (7일 또는 28일 기준).")
    A("")
    A("## 한눈에")
    for m in R["markets"]:
        ok = [r for r in m["setups"] if r["verdict"] == "통과"]
        weak = [r for r in m["setups"] if r["verdict"].startswith("약함")]
        bad = [r for r in m["setups"] if r["verdict"].startswith("반대")]
        real = [r for r in m["setups"] if not r["id"].startswith("rand")]
        A(f"### {m['label']} ({m['coins']}종목)")
        A(f"- 아무 코인이나 아무 날에 샀을 때 7일 실수익: 고르는 기간 {pc(m['base']['net_train'])}, 확인 기간 {pc(m['base']['net_test'])}")
        A(f"- 시나리오 {len(real)}개 중 통과 {len(ok)}개, 약함 {len(weak)}개, 반대 {len(bad)}개")
        for r in ok + weak:
            b = r["test"]
            A(f"  - [{r['verdict'].split(' ')[0]}] {r['group']} — {r['label'].strip()}: 확인 기간 {b['n']}건, 7일 {pc(b['net'])} / 초과 {ext(b)}, 28일 {pc(b.get('net28'))} / 초과 {ext(b, 't28', 'ex28')}")
        A("")

    for m in R["markets"]:
        A(f"## {m['label']}")
        A("")
        A("### A. 단계형 시나리오")
        A("| 시나리오 | 건수 (고름/확인) | 급등 배수 | 7일 실수익 (고름/확인) | 확인 중앙값·이긴 비율 | 7일 초과 고름 (t) | 7일 초과 확인 (t) | 28일 실수익 (고름/확인) | 28일 초과 확인 (t) | 손절·2배 익절 매매 (고름/확인, 평균 R) | 추세 따라가기 매매 (고름/확인) | 판정 |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|")
        g0 = None
        for r in m["setups"]:
            a, b = r["train"], r["test"]
            if r["group"] != g0:
                g0 = r["group"]
                A(f"| **{g0}** | | | | | | | | | | | |")
            if a.get("n", 0) < 30 or b.get("n", 0) < 30:
                A(f"| {r['label']} | {a.get('n', 0)} / {b.get('n', 0)} | - | - | - | - | - | - | - | - | - | 건수 부족 |")
                continue
            ea, eb = a.get("exits") or {}, b.get("exits") or {}
            r2 = f"{pc((ea.get('r2') or {}).get('mean'))} / {pc((eb.get('r2') or {}).get('mean'))}, {num((eb.get('r2') or {}).get('R'), 2)}R"
            tr = f"{pc((ea.get('trail') or {}).get('mean'))} / {pc((eb.get('trail') or {}).get('mean'))}"
            A(f"| {r['label']} | {a['n']} / {b['n']} | {num(a['lift'], 2)} / {num(b['lift'], 2)} | {pc(a['net'])} / {pc(b['net'])} | {pc(b['median'])} · {round(b['win'] * 100)}% | {ext(a)} | {ext(b)} | {pc(a.get('net28'))} / {pc(b.get('net28'))} | {ext(b, 't28', 'ex28')} | {r2} | {tr} | {r['verdict']} |")
        A("")

        A("### B. 자료에서 찾은 조건 조합 (2023년까지로 나무를 키우고 2024년 이후에 그대로 적용)")
        for key, title in (("ex7", "7일 초과수익 기준"), ("ex28", "28일 초과수익 기준"), ("abs7", "7일 실수익 기준 (시장 상태 지표 포함)"), ("new_only", "주봉·순서형 지표만으로, 7일 초과수익 기준")):
            mn = (m.get("mines") or {}).get(key)
            if not mn:
                continue
            A(f"**{title}** — 조합 {mn['leaves']}개 중 고르는 기간 성적 상위 5개와 하위 3개")
            A("")
            A("| 조건 조합 | 고르는 기간 건수 · 초과 · 실수익 | 확인 기간 건수 · 초과 (t) · 실수익 · 중앙값 · 이긴 비율 |")
            A("|---|---|---|")
            for tag, rows in (("상위", mn["best"]), ("하위", mn["worst"])):
                for r in rows:
                    a, b = r["train"], r["test"]
                    if "ex" not in a:
                        continue
                    tb = f"{b['n']} · {ext(b)} · {pc(b['net'])} · {pc(b['median'])} · {round(b['win'] * 100)}%" if "ex" in b else f"{b.get('n', 0)}건 (부족)"
                    A(f"| [{tag}] {' → '.join(r['rule'])} | {a['n']} · {pp(a['ex'])} · {pc(a['net'])} | {tb} |")
            A("")

        md = m.get("models")
        if md and md["rows"]:
            A("### C. 모델 재학습 (일봉 지표 103개 + 주봉·순서형 지표)")
            A("| 모델 | 기간 | 급등 적중 (기준) | 거래당 실수익 | 중앙값 | 초과 (t) | 1차 모델(일봉만): 급등 적중 · 실수익 · 초과 (t) |")
            A("|---|---|---|---|---|---|---|")
            for r in md["rows"]:
                for per, nm in (("train", "~2023"), ("test", "2024~")):
                    p = r.get(per) or {}
                    if "net" not in p:
                        continue
                    b = (r.get("baseline") or {}).get(per)
                    bt = f"{pc(b['pump'])} · {pc(b['net'])} · {pp(b['ex'])} ({num(b['t'])})" if b else "-"
                    A(f"| {r['label']} | {nm} | {pc(p['pump'])} ({pc(p['base'])}) | {pc(p['net'])} | {pc(p['median'])} | {ext(p)} | {bt} |")
            A("")
            for r in md["rows"]:
                A(f"**{r['label']}이 많이 기댄 지표** (★ = 이번에 더한 주봉·순서형 지표): " + ", ".join(("★" if i["new"] else "") + i["label"] for i in r["importance"][:12]))
                A("")
                A(f"2024년~ 가장 자주 뽑힌 종목: " + ", ".join(f"{k} {v}" for k, v in r["topPicks"].items()))
                A("")

        nw = m.get("news")
        if nw:
            A("### D. 뉴스 × 차트")
            A(f"코인이 잡힌 뉴스 제목 {nw['titles']:,}건 → 뉴스가 붙은 코인·날 {nw['eventDays']:,}개 ({nw['coins']}종목). 뉴스가 나온 날 종가에 판단해 다음 날 시가에 산 경우.")
            A("")
            A("| 뉴스 종류 | 건수 (고름/확인) | 뉴스 당일 등락 (중앙값) | 급등 배수 | 7일 실수익 (고름/확인) | 7일 초과 고름 (t) | 7일 초과 확인 (t) | 28일 실수익 (고름/확인) | 28일 초과 확인 (t) |")
            A("|---|---|---|---|---|---|---|---|---|")
            for r in nw["cats"]:
                a, b = r["train"], r["test"]
                if a.get("n", 0) < 30 or b.get("n", 0) < 30:
                    A(f"| {r['label']} | {a.get('n', 0)} / {b.get('n', 0)} | - | - | - | - | - | - | - |")
                    continue
                A(f"| {r['label']} | {a['n']} / {b['n']} | {pc(r.get('roc1'))} | {num(a['lift'], 2)} / {num(b['lift'], 2)} | {pc(a['net'])} / {pc(b['net'])} | {ext(a)} | {ext(b)} | {pc(a.get('net28'))} / {pc(b.get('net28'))} | {ext(b, 't28', 'ex28')} |")
            A("")
            A("**호재성 뉴스가 나온 날, 그때의 차트 상태별**")
            A("")
            A("| 차트 상태 | 건수 (고름/확인) | 7일 실수익 (고름/확인) | 확인 중앙값·이긴 비율 | 7일 초과 고름 (t) | 7일 초과 확인 (t) | 28일 실수익 (고름/확인) |")
            A("|---|---|---|---|---|---|---|")
            for r in nw["states"]:
                a, b = r["train"], r["test"]
                if a.get("n", 0) < 30 or b.get("n", 0) < 30:
                    A(f"| {r['label']} | {a.get('n', 0)} / {b.get('n', 0)} | - | - | - | - | - |")
                    continue
                A(f"| {r['label']} | {a['n']} / {b['n']} | {pc(a['net'])} / {pc(b['net'])} | {pc(b['median'])} · {round(b['win'] * 100)}% | {ext(a)} | {ext(b)} | {pc(a.get('net28'))} / {pc(b.get('net28'))} |")
            A("")
            mn = nw.get("tree")
            if mn:
                A("**호재성 뉴스가 나온 날만 놓고 찾은 조건 조합**")
                A("")
                A("| 조건 조합 | 고르는 기간 건수 · 초과 · 실수익 | 확인 기간 건수 · 초과 (t) · 실수익 · 중앙값 · 이긴 비율 |")
                A("|---|---|---|")
                for tag, rows in (("상위", mn["best"]), ("하위", mn["worst"])):
                    for r in rows:
                        a, b = r["train"], r["test"]
                        if "ex" not in a:
                            continue
                        tb = f"{b['n']} · {ext(b)} · {pc(b['net'])} · {pc(b['median'])} · {round(b['win'] * 100)}%" if "ex" in b else f"{b.get('n', 0)}건 (부족)"
                        A(f"| [{tag}] {' → '.join(r['rule'])} | {a['n']} · {pp(a['ex'])} · {pc(a['net'])} | {tb} |")
                A("")

    A("## 읽을 때 주의")
    A("- 시나리오 30여 개와 나무가 찾은 조합 수십 개를 한꺼번에 봤다. 이 정도면 우연히 좋아 보이는 것이 한두 개는 나오므로, 두 기간·두 거래소에서 같은 방향인지를 본다.")
    A("- 주봉은 달력 주가 아니라 '그날로 끝나는 7일'로 만들었다. 피벗(고점·저점)은 양옆 3일을 봐야 확정되므로 3일 늦게 반영했다.")
    A("- 구조 기반 매매는 장중에 손절가·목표가에 닿으면 그 가격(갭이면 시가)에 체결됐다고 가정했고, 같은 날 둘 다 닿으면 손절이 먼저라고 봤다.")
    A("- 뉴스는 제목에 코인 이름이나 티커가 들어간 기사만 잡았고, 종류는 제목의 단어로 나눴다. 기사가 가격이 움직인 뒤에 나오는 경우가 많다는 점(뉴스 당일 등락 칸)을 같이 봐야 한다.")
    A("- 빗썸은 지금 상장된 종목만 들어 있다.")
    return "\n".join(L) + "\n"
