#!/usr/bin/env python3
"""
5차 검증 — "오르기 직전 저점에서 살 수 있었나"를 실제로 크게 오른 구간에서 거꾸로 확인한다.

  A. 큰 상승 구간 찾기: 종가가 앞뒤 15일 중 가장 낮은 날(저점)에서 45일 안에 고가 기준 2배 이상 오른 구간.
     저점일의 BTC 상태로 'BTC 동반'(BTC가 50일선 위이고 28일간 올랐음)과 '독자 상승'으로 나눈다.
  B. 그 구간마다 신호가 언제 처음 떴는지: 저점 며칠 뒤, 저점보다 몇 % 위에서, 그때 사면 고점까지 얼마가 남았는지.
     같은 신호를 평소에 그대로 따라 샀을 때의 성적(헛신호 포함)과 나란히 본다.
  C. 저점을 실시간으로 알 수 있는 가장 빠른 시점(20일 최저가를 찍고 3일간 안 깨져 저점이 확정된 날)에 샀다면.
     크게 오른 저점과 그렇지 않은 저점을 가르는 특징이 있는지.
  D. 사례: 지정 종목의 2025년 이후 큰 상승 구간별 경과.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "ml4"), os.path.join(HERE, "..", "ml3"), os.path.join(HERE, "..", "ml")):
    sys.path.insert(0, p)
import study as S  # noqa: E402
import study3 as T  # noqa: E402
import study4 as P4  # noqa: E402

OUT = S.OUT
log = S.log
swv = np.lib.stride_tricks.sliding_window_view
WIN, HALF, LOOK = 45, 15, 30
CASES = ["XLM", "SUI", "ZEC", "NEAR", "HYPE", "PUMP", "UNI"]


def find_rallies(c, min_gain):
    """저점 = 종가가 앞뒤 15일 중 최저 (사후에만 알 수 있음 — 구간을 표시하는 용도로만 쓴다)"""
    cl, hi = c["c"], c["h"]
    n = len(cl)
    out = []
    if n < 2 * HALF + 10:
        return out
    mn = np.full(n, np.nan)
    mn[HALF : n - HALF] = swv(cl, 2 * HALF + 1).min(axis=1)
    last = -999
    for i in np.flatnonzero(cl == mn):
        if i - last < 40 or i + 5 >= n:
            continue
        e = min(i + WIN, n - 1)
        pk = i + 1 + int(np.argmax(hi[i + 1 : e + 1]))
        gain = hi[pk] / cl[i] - 1
        if gain >= min_gain:
            out.append((int(i), pk, float(gain)))
            last = i
    return out


def signals(df):
    g = lambda k: np.nan_to_num(df[k].to_numpy())
    vr, roc1 = df.vr.to_numpy(), df.roc1.to_numpy()
    return {
        "vol3": ("거래량 3배 이상인 날", vr >= 3),
        "up10": ("하루 +10% 이상 오른 날", roc1 >= 0.10),
        "brk20v": ("20일 고점 돌파 + 거래량 2배", (g("brk20") == 1) & (vr >= 2)),
        "base": ("횡보 뒤 60일 고점 돌파 + 거래량 2배", (df.range60_prev.to_numpy() < 0.6) & (g("brk60") == 1) & (vr >= 2)),
        "turn": ("저점 높임 확정 뒤 직전 고점 돌파", (g("turn10") == 1) & (g("brk_piv") == 1)),
        "re50": ("50일선 회복", g("reclaim50") == 1),
        "gc": ("5·20일 골든크로스", g("gc_5_20") == 1),
        "wmacd": ("주봉 MACD 골든크로스", g("w_macd_cross") == 1),
        "brk250": ("250일 고점 돌파", g("brk250") == 1),
    }


def rally_study(df, coins, tagged, tz):
    idx_map = {}
    bars = df.bar.to_numpy()
    for sym, idx in df.groupby("sym").indices.items():
        a = np.full(len(coins[sym]["c"]), -1)
        a[bars[idx]] = idx
        idx_map[sym] = a
    SG = signals(df)
    news = None
    if tagged is not None:
        tg = tagged.copy()
        tg["date"] = (tg.ts + pd.Timedelta(hours=tz)).dt.strftime("%Y-%m-%d")
        good = tg.cats.str.contains("partner|upgrade|inst|burn|fund")
        news = (set(zip(tg.sym, tg.date)), set(zip(tg.sym[good], tg.date[good])))
    btc_up = (df.btc_d_sma50.to_numpy() > 0) & (df.btc_roc28.to_numpy() > 0)
    rows = []
    for sym, a in idx_map.items():
        c = coins[sym]
        for i0, pk, gain in find_rallies(c, 0.6):
            r0 = a[i0]
            if r0 < 0:
                continue  # 거래대금 기준 미달 등으로 대상이 아니던 날
            dates = c["dates"]
            r = dict(sym=sym, date=dates[i0], peak_date=dates[pk], days=pk - i0, gain=gain, low=float(c["c"][i0]), high=float(c["h"][pk]), btc_up=bool(btc_up[r0]), btc_rsi=float(df.btc_rsi14.to_numpy()[r0]), test=bool(df.is_test.to_numpy()[r0]),
                     drop30=float(c["c"][i0] / c["c"][i0 - 30] - 1) if i0 >= 30 else None, sig={})
            end = min(pk - 1, i0 + LOOK, len(dates) - 2)
            for k in range(0, end - i0 + 1):
                rk = a[i0 + k]
                if rk < 0:
                    continue
                for sid, (_, m) in SG.items():
                    if sid not in r["sig"] and m[rk]:
                        entry = c["o"][i0 + k + 1]
                        r["sig"][sid] = dict(k=k, above=float(c["c"][i0 + k] / c["c"][i0] - 1), left=float(c["h"][pk] / entry - 1), part=float((c["h"][pk] - entry) / (c["h"][pk] - c["c"][i0])))
            if news:
                near = lambda lo, hi_, s: sum((sym, dates[j]) in s for j in range(max(i0 + lo, 0), min(i0 + hi_, len(dates) - 1) + 1))
                r["news_before"], r["news_early"] = near(-7, 0, news[0]), near(1, 5, news[0])
                r["good_before"], r["good_early"] = near(-7, 0, news[1]), near(1, 5, news[1])
            rows.append(r)
    return rows, SG


def summarize(rows, SG, df):
    def block(rs):
        if len(rs) < 20:
            return dict(n=len(rs))
        out = dict(n=len(rs), median_gain=float(np.median([r["gain"] for r in rs])), median_days=float(np.median([r["days"] for r in rs])), sig={})
        first = []
        for r in rs:
            if r["sig"]:
                b = min(r["sig"].values(), key=lambda x: x["k"])
                first.append(b)
        out["any"] = dict(catch=len(first) / len(rs), k=float(np.median([x["k"] for x in first])) if first else None, above=float(np.median([x["above"] for x in first])) if first else None, part=float(np.median([x["part"] for x in first])) if first else None)
        for sid in SG:
            hit = [r["sig"][sid] for r in rs if sid in r["sig"]]
            out["sig"][sid] = dict(catch=len(hit) / len(rs), k=float(np.median([x["k"] for x in hit])) if hit else None, above=float(np.median([x["above"] for x in hit])) if hit else None, left=float(np.median([x["left"] for x in hit])) if hit else None, part=float(np.median([x["part"] for x in hit])) if hit else None)
        if "news_before" in rs[0]:
            out["news"] = dict(before=float(np.mean([r["news_before"] > 0 for r in rs])), early=float(np.mean([r["news_early"] > 0 for r in rs])), good_before=float(np.mean([r["good_before"] > 0 for r in rs])), good_early=float(np.mean([r["good_early"] > 0 for r in rs])))
        return out

    big = [r for r in rows if r["gain"] >= 1.0]
    groups = {}
    for per, pt in (("train", False), ("test", True)):
        for gname, fn in (("all", lambda r: True), ("solo", lambda r: not r["btc_up"]), ("with_btc", lambda r: r["btc_up"])):
            groups[f"{per}_{gname}"] = block([r for r in big if r["test"] == pt and fn(r)])
    # 같은 신호를 평소에 그대로 따라 샀을 때 (헛신호 포함)
    base = {sid: dict(label=lab, **P4.dist_stats(df, T.dedup(df, np.asarray(m, bool)))) for sid, (lab, m) in SG.items()}
    years = {}
    for r in big:
        y = r["date"][:4]
        d = years.setdefault(y, dict(n=0, solo=0))
        d["n"] += 1
        d["solo"] += int(not r["btc_up"])
    return dict(groups=groups, base=base, years=years)


def pivot_lows(df, coins):
    """20일 최저가를 찍고 3일간 그 저가가 안 깨져 '저점 확정'이 된 날"""
    m = np.zeros(len(df), bool)
    bars = df.bar.to_numpy()
    for sym, idx in df.groupby("sym").indices.items():
        l = coins[sym]["l"]
        n = len(l)
        if n < 30:
            continue
        L = pd.Series(l)
        is_low = (L == L.rolling(20).min()).to_numpy()
        ok = np.zeros(n, bool)
        for i in range(3, n):
            j = i - 3
            ok[i] = is_low[j] and l[j + 1] > l[j] and l[j + 2] > l[j] and l[j + 3] > l[j]
        m[idx] = ok[bars[idx]]
    return m


# ----------------------------------------------------------------------------------------
pc = P4.pc
sh = P4.sh


def render(R):
    L = []
    A = L.append
    A("# 크게 오른 구간을 거꾸로 따라가 보기 — 오르기 직전에 살 수 있었나")
    A("")
    A(f"생성 {R['generatedAt'].replace('T', ' ')} UTC · 큰 상승 구간 = 저점(종가가 앞뒤 15일 중 최저)에서 45일 안에 고가 기준 2배 이상 · 고르는 기간 ~2023 / 확인 기간 2024~")
    A("")
    for m in R["markets"]:
        A(f"## {m['label']}")
        A("")
        A("연도별 큰 상승 구간 수 (그중 BTC가 약할 때 혼자 오른 것): " + " / ".join(f"{y} {v['n']}개 ({v['solo']})" for y, v in sorted(m["summary"]["years"].items())))
        A("")
        A("### B. 신호는 저점 며칠 뒤, 얼마나 올라온 뒤에 떴나")
        A("잡은 비율 = 저점부터 고점 전까지(최대 30일) 그 신호가 한 번이라도 뜬 구간의 비율. 남은 몫 = 신호 다음 날 시가에 샀을 때, 저점→고점 상승폭 중 아직 남아 있던 부분.")
        A("")
        for key, title in (("test_all", "2024년 이후 전체"), ("test_solo", "2024년 이후, BTC가 약할 때 혼자 오른 구간"), ("test_with_btc", "2024년 이후, BTC 상승과 함께 오른 구간"), ("train_all", "2023년까지 전체"), ("train_solo", "2023년까지, 혼자 오른 구간")):
            G = m["summary"]["groups"].get(key)
            if not G or "sig" not in G:
                continue
            A(f"**{title}** — {G['n']}개, 상승폭 중앙값 {pc(G['median_gain'], 0)}, 저점에서 고점까지 {G['median_days']:.0f}일")
            if G.get("news"):
                nw = G["news"]
                A(f"- 뉴스: 저점 전 7일 안에 그 코인 기사가 있던 구간 {sh(nw['before'])} (호재성 {sh(nw['good_before'])}), 저점 뒤 5일 안 {sh(nw['early'])} (호재성 {sh(nw['good_early'])})")
            a = G["any"]
            A(f"- 아래 신호 중 가장 빠른 것 기준: 잡은 비율 {sh(a['catch'])}, 저점 {a['k']:.0f}일 뒤, 저점보다 {pc(a['above'], 0)} 위, 남은 몫 {sh(a['part'])}")
            A("")
            A("| 신호 | 잡은 비율 | 저점 며칠 뒤 | 저점보다 얼마 위 | 그때 사서 고점까지 | 남은 몫 | 같은 신호를 평소에 다 따라 샀다면: 28일 평균 · 중앙값 · −20% 이하 비율 |")
            A("|---|---|---|---|---|---|---|")
            per = "test" if key.startswith("test") else "train"
            for sid, s in G["sig"].items():
                b = m["summary"]["base"][sid]
                bb = b.get(per, {})
                bt = f"{pc(bb.get('net28'))} · {pc(bb.get('med28'))} · {sh(bb.get('loss'))} ({bb.get('n', 0)}건)" if "net28" in bb else "-"
                if s["k"] is None:
                    A(f"| {b['label']} | {sh(s['catch'])} | - | - | - | - | {bt} |")
                else:
                    A(f"| {b['label']} | {sh(s['catch'])} | {s['k']:.0f}일 | {pc(s['above'], 0)} | {pc(s['left'], 0)} | {sh(s['part'])} | {bt} |")
            A("")
        P = m.get("pivot")
        if P:
            A("### C. 저점이 확정된 날(20일 최저가 뒤 3일간 안 깨짐)에 샀다면")
            d = P["dist"]
            if "net28" in d.get("train", {}) and "net28" in d.get("test", {}):
                a, b = d["train"], d["test"]
                A(f"건수 {a['n']} / {b['n']} · 28일 평균 {pc(a['net28'])} / {pc(b['net28'])} · 중앙값 {pc(a['med28'])} / {pc(b['med28'])} · 28일 +50% 이상 {sh(a['big'])} / {sh(b['big'])} · −20% 이하 {sh(a['loss'])} / {sh(b['loss'])} · 28일 안 최고점 평균 {pc(a['peak'], 0)} / {pc(b['peak'], 0)}")
                A("")
            for k2, t2 in (("winners", "크게 오른 저점(28일 +50% 이상)과 나머지를 가르는 특징"),):
                if P.get(k2):
                    A(f"**{t2}** — 가르는 힘 50%는 구분 못 함")
                    A("")
                    A("| 특징 | 크게 오른 저점의 중앙값 (고름/확인) | 나머지 (고름/확인) | 가르는 힘 |")
                    A("|---|---|---|---|")
                    for r in P[k2][:10]:
                        a, b = r["train"], r["test"]
                        A(f"| {r['label']} | {P4.fmt(a['hit'])} / {P4.fmt(b['hit'])} | {P4.fmt(a['rest'])} / {P4.fmt(b['rest'])} | {sh(max(a['auc'], 1 - a['auc']))} / {sh(max(b['auc'], 1 - b['auc']))} ({'클수록' if a['auc'] > 0.5 else '작을수록'}) |")
                    A("")
            if P.get("tree"):
                A("**조건 조합** (고르는 기간에서 찾아 확인 기간에 적용)")
                A("")
                A("| 조건 조합 | 고르는 기간: 건수 · 28일 평균 · 중앙값 · +50% 이상 | 확인 기간: 건수 · 28일 평균 · 중앙값 · +50% 이상 · −20% 이하 |")
                A("|---|---|---|")
                for r in P["tree"]:
                    a, b = r["train"], r["test"]
                    ta = f"{a['n']} · {pc(a['net28'])} · {pc(a['med28'])} · {sh(a['big'])}" if "net28" in a else f"{a['n']}건"
                    tb = f"{b['n']} · {pc(b['net28'])} · {pc(b['med28'])} · {sh(b['big'])} · {sh(b['loss'])}" if "net28" in b else f"{b['n']}건"
                    A(f"| {' → '.join(r['rule'])} | {ta} | {tb} |")
                A("")
        for title, rows in (("D. 사례 — 지정 종목의 2025년 이후 상승 구간 (+60% 이상)", m.get("cases") or []), ("D'. 2025년 이후 BTC가 약할 때 혼자 가장 크게 오른 구간 20개", m.get("topSolo") or [])):
            if not rows:
                continue
            A(f"### {title}")
            A("| 종목 | 저점일 → 고점일 | 상승폭 | 저점 직전 30일 | 저점일 BTC | 저점 전 7일 / 뒤 5일 기사 | 가장 빠른 신호 (저점 며칠 뒤 · 저점보다 얼마 위 · 남은 몫) | 횡보 뒤 돌파 신호 |")
            A("|---|---|---|---|---|---|---|---|")
            for r in rows:
                sg = r["sig"]
                first = min(sg.items(), key=lambda kv: kv[1]["k"]) if sg else None
                ft = f"{m['sigLabels'][first[0]]} ({first[1]['k']}일 · {pc(first[1]['above'], 0)} · {sh(first[1]['part'])})" if first else "없음"
                bs = sg.get("base")
                bt = f"{bs['k']}일 뒤 · {pc(bs['above'], 0)} 위 · 남은 몫 {sh(bs['part'])}" if bs else "안 뜸"
                nw = f"{r.get('news_before', '-')} / {r.get('news_early', '-')}"
                A(f"| {r['sym']} | {r['date']} → {r['peak_date']} ({r['days']}일) | {pc(r['gain'], 0)} | {pc(r['drop30'], 0)} | {'강함' if r['btc_up'] else '약함'} (RSI {r['btc_rsi']:.0f}) | {nw} | {ft} | {bt} |")
            A("")
    A("## 읽을 때 주의")
    A("- 저점일은 나중에 돌아봐야 알 수 있는 날이다. 이 표는 '그때 무엇이 보였나'를 확인하는 용도이고, B의 오른쪽 칸(같은 신호를 평소에 다 따라 샀을 때)이 실제로 그 신호로 매매했을 때의 성적이다.")
    A("- 뉴스는 한국 매체 한 곳의 제목에 코인 이름이 들어간 것만 센 것이라, 기사가 없다고 재료가 없었다는 뜻은 아니다.")
    A("- 거래대금 기준에 못 미치던 날이 저점인 구간은 뺐다.")
    return "\n".join(L) + "\n"


def main():
    t0 = time.time()
    raw = {m: S.load_market(m, cfg) for m, cfg in S.MARKETS.items() if os.path.isdir(f"{S.DATA}/{m}")}
    news = T.load_news()
    tagged = None
    if news is not None:
        names = {}
        if os.path.isdir(f"{S.DATA}/bithumb"):
            for fn in os.listdir(f"{S.DATA}/bithumb"):
                with open(f"{S.DATA}/bithumb/{fn}", encoding="utf-8") as f:
                    j = json.load(f)
                names[j["sym"]] = (j.get("name"), j.get("en"))
        for m in raw:
            for s in raw[m]:
                names.setdefault(s, (None, None))
        tagged = T.tag_news(news, names)
        log(f"뉴스 제목 {len(news):,}건 중 코인이 잡힌 것 {len(tagged):,}건")
    result = dict(generatedAt=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M"), markets=[])
    for mkt in ("binance",):
        cfg = S.MARKETS[mkt]
        if mkt not in raw or "BTC" not in raw[mkt]:
            continue
        raw.setdefault("bithumb", {})
        log(f"[{cfg['label']}] 지표 계산")
        df, feats, all_dates, _ = S.build(mkt, cfg, raw)
        df, _ = T.add_features3(df, raw[mkt])
        df = P4.add_features4(df, raw[mkt])
        rows, SG = rally_study(df, raw[mkt], tagged, cfg["tz"])
        log(f"  상승 구간 {len(rows):,}개 (+60% 이상), 그중 2배 이상 {sum(r['gain'] >= 1 for r in rows):,}개")
        summ = summarize(rows, SG, df)
        pm = T.dedup(df, pivot_lows(df, raw[mkt]))
        piv = P4.profile(df, pm, raw[mkt], cfg["cost"])
        log(f"  저점 확정일 {int(pm.sum()):,}건 분석 완료")
        cases = sorted([r for r in rows if r["sym"] in CASES and r["date"] >= "2025-01-01"], key=lambda r: (CASES.index(r["sym"]), r["date"]))
        top_solo = sorted([r for r in rows if r["date"] >= "2025-01-01" and not r["btc_up"]], key=lambda r: -r["gain"])[:20]
        result["markets"].append(dict(market=mkt, label=cfg["label"], summary=summ, pivot=dict(dist=piv["dist"], winners=piv["winners"], tree=piv["tree"], stats=piv["stats"]), cases=cases, topSolo=top_solo, sigLabels={k: v[0] for k, v in SG.items()},
                                      missing=[s for s in CASES if s not in raw[mkt]]))

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else round(float(o), 5)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    result = clean(result)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/rallies.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(f"{OUT}/RALLIES.md", "w", encoding="utf-8") as f:
        f.write(render(result))
    log(f"끝 ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
