#!/usr/bin/env python3
"""리포트 `terms` 검사 (REPORT_GUIDE.md 7-4). 사용: python3 prompts/check_terms.py data/reports/2026-11-09-ena.json [...]
앱(lib/terms.ts)과 같은 방식으로 표기를 접어 비교한다: 대소문자·띄어쓰기·하이픈·가운뎃점·따옴표·괄호 무시.
문제가 있으면 목록을 찍고 종료 코드 1."""
import json, re, sys, unicodedata

KEY_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
FOLD = re.compile(r"[\s\-–—_.·・'’\"“”()]")
SKIP = {"id", "date", "slot", "project", "ticker", "question", "meta", "sources", "terms", "openItems", "glossary"}


def fold(s: str) -> str:
    return FOLD.sub("", unicodedata.normalize("NFKC", s).lower())


def body_texts(v, key=None):
    if key in SKIP:
        return
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for x in v:
            yield from body_texts(x)
    elif isinstance(v, dict):
        for k, x in v.items():
            yield from body_texts(x, k)


def check(path: str) -> list[str]:
    r = json.load(open(path, encoding="utf-8"))
    if not isinstance(r, dict):
        return ["리포트 파일이 아님 (index.json 등은 넣지 않는다)"]
    terms = r.get("terms") or []
    body = fold(" ".join(body_texts(r)))
    errs = []
    if len(terms) < 5:
        errs.append(f"terms {len(terms)}개 — 5개 이상")
    keys, owner = set(), {}
    inline = 0
    for i, t in enumerate(terms):
        name = t.get("term", f"#{i}")
        for f in ("term", "key", "meaning"):
            if not isinstance(t.get(f), str) or not t[f].strip():
                errs.append(f"{name}: {f} 없음")
        if not isinstance(t.get("inline"), bool):
            errs.append(f"{name}: inline(true/false) 없음")
        k = t.get("key", "")
        if isinstance(k, str) and k:
            if not KEY_RE.match(k):
                errs.append(f"{name}: key '{k}' 형식 (영문 소문자·숫자·하이픈)")
            if k in keys:
                errs.append(f"{name}: key '{k}' 중복")
            keys.add(k)
        variants = [name] + [a for a in (t.get("aliases") or []) if isinstance(a, str)]
        parts = [p.strip() for v in variants for p in v.split("·") if p.strip()]
        for p in parts:
            fp = fold(p)
            if fp in owner and owner[fp] != i:
                errs.append(f"{name}: 표기 '{p}'가 다른 항목({terms[owner[fp]].get('term')})과 겹침")
            owner[fp] = i
        if t.get("inline") is True:
            inline += 1
            if not any(fold(p) and fold(p) in body for p in parts):
                errs.append(f"{name}: inline인데 본문에 term·aliases 어느 표기로도 나오지 않음")
            src = t.get("sources") or []
            if not any(isinstance(s, dict) and str(s.get("url", "")).startswith("http") for s in src):
                errs.append(f"{name}: inline인데 sources 없음")
        if "role" in t and (not isinstance(t["role"], str) or not t["role"].strip()):
            errs.append(f"{name}: role이 비어 있으면 필드를 빼라")
    if not 6 <= inline <= 14:
        errs.append(f"inline {inline}개 — 6~14개")
    return errs


if __name__ == "__main__":
    bad = False
    for p in sys.argv[1:]:
        errs = check(p)
        print(f"{p}: {'OK' if not errs else str(len(errs)) + '건'}")
        for e in errs:
            print("  -", e)
        bad |= bool(errs)
    sys.exit(1 if bad else 0)
