# Coin Cockpit — Data

Coin Cockpit 앱이 읽는 분석 데이터 저장소. 예약 작업(Claude)이 자동으로 커밋한다.

```
data/reports/            코인 심층 리포트 (월·수·금)   → YYYY-MM-DD-ticker.json + index.json
data/daily/              데일리 시장 브리프 (매일 09:00) → YYYY-MM-DD.json + index.json
schema/                  JSON 스키마
prompts/REPORT_GUIDE.md  리포트 작성 지침
prompts/DAILY_GUIDE.md   데일리 작성 지침 (인과관계 규칙)
```

앱은 `https://raw.githubusercontent.com/<owner>/<repo>/main/data/...` 에서 읽는다.
이 저장소에는 분석 텍스트만 들어가며 API 키·잔고 등 개인 정보는 절대 포함하지 않는다.
