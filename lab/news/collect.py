#!/usr/bin/env python3
"""
뉴스 제목 수집 — 블록미디어의 공개 워드프레스 API에서 기사 번호·게시 시각·제목만 받는다.

  - 본문은 받지 않는다. 받은 제목은 작업 캐시에만 두고 저장소에는 올리지 않는다 (월별 건수만 올린다).
  - 한 달씩 끊어서 받고, 이미 받은 지난 달은 다시 받지 않는다. 요청 사이에 쉬어 간다.
  - 시간 예산을 넘기면 거기서 멈추고, 다음 실행 때 이어서 받는다.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

OUT = os.path.join(os.environ.get("LAB_NEWS", "lab/.news"), "blockmedia")
META = os.environ.get("LAB_OUT", "lab/results") + "/news-meta.json"
BASE = "https://www.blockmedia.co.kr/wp-json/wp/v2/posts"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
START = os.environ.get("NEWS_START", "2019-01")
BUDGET = float(os.environ.get("NEWS_BUDGET_SEC", "4200"))
PAUSE = 0.7


def get(params):
    url = BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8")), int(r.headers.get("X-WP-TotalPages") or 0)
        except urllib.error.HTTPError as e:
            if e.code == 400:  # 마지막 쪽을 넘어감
                return [], 0
            if e.code in (429, 500, 502, 503, 504, 520, 522, 524):
                time.sleep(10 * (attempt + 1))
                continue
            raise
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("요청 실패가 이어져 멈춤")


def months(start):
    y, m = map(int, start.split("-"))
    now = time.gmtime()
    while (y, m) <= (now.tm_year, now.tm_mon):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    now = time.gmtime()
    done = stopped = 0
    for y, m in months(START):
        path = f"{OUT}/{y}-{m:02d}.json"
        recent = (y, m) >= ((now.tm_year, now.tm_mon - 1) if now.tm_mon > 1 else (now.tm_year - 1, 12))
        if os.path.exists(path) and not recent:
            continue
        if time.time() - t0 > BUDGET:
            stopped = 1
            break
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        rows, page = [], 1
        try:
            while True:
                data, pages = get({"per_page": 100, "page": page, "_fields": "id,date_gmt,title", "orderby": "date", "order": "asc", "after": f"{y}-{m:02d}-01T00:00:00", "before": f"{ny}-{nm:02d}-01T00:00:00"})
                rows += [[p["id"], p["date_gmt"], (p.get("title") or {}).get("rendered", "")] for p in data]
                time.sleep(PAUSE)
                if len(data) < 100 or (pages and page >= pages):
                    break
                page += 1
        except Exception as e:
            print(f"{y}-{m:02d}: 중단 ({e})", flush=True)
            stopped = 1
            break
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
        done += 1
        print(f"{y}-{m:02d}: {len(rows)}건 ({page}쪽) · 경과 {time.time() - t0:.0f}초", flush=True)
    counts = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.endswith(".json"):
            with open(f"{OUT}/{fn}", encoding="utf-8") as f:
                counts[fn[:-5]] = len(json.load(f))
    os.makedirs(os.path.dirname(META), exist_ok=True)
    with open(META, "w", encoding="utf-8") as f:
        json.dump(dict(source="blockmedia.co.kr 공개 API (제목·시각만)", updated=time.strftime("%Y-%m-%dT%H:%M", time.gmtime()), complete=not stopped, total=sum(counts.values()), months=counts), f, ensure_ascii=False, indent=1)
    print(f"이번에 받은 달 {done}개 · 누적 {sum(counts.values()):,}건 · {'시간 예산으로 중단, 다음에 이어받음' if stopped else '전부 받음'}")


if __name__ == "__main__":
    main()
