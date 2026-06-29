# 책소개 문구가 책을 팔까? — 알라딘 SalesPoint 분석

차트인한 한국 책 1,814권에서 **책소개 copy·장르·길이·가격**이 판매지수(알라딘 SalesPoint)와
어떤 관계인지 규명하고 예측. **결론(정직 음성): copy는 판매 모멘텀을 거의 못 움직임 — 장르·누적 리뷰가 지배.**

## 핵심 결과
- 책소개 copy 자기 몫 ΔR² = **+0.008** (합격선 0.02 미달), 유의 텍스트 feature 1/9.
- "베스트셀러/수상 키워드 2× 효과"는 **장르 교란** — 통제하면 유의 0/8 (역인과/착시).
- 사전런칭 예측은 장르평균을 거의 못 넘음 (20시드 0/20이 +0.05 도달).
- 두 합격선 모두 **NOT MET** — robust(다중시드·비선형·매개 점검 통과).

## 파이프라인 (9 stage)
`docs/00`~`06` = 단계별 결정 로그. 산출물: `analysis.ipynb`, `report.html`.

## 재현
```bash
# 1. 알라딘 TTBKey를 .env에 (ALADIN_TTBKEY=...)
uv run python src/collect.py        # 수집 → data/raw/corpus.csv (cache 재사용)
uv run python src/data_quality.py data/raw/corpus.csv --key isbn13   # Stage4 profile
uv run python src/prepare.py        # → data/model_table.csv (+ feature_groups.json)
uv run python src/data_quality.py data/model_table.csv --strict --key isbn13   # Stage6 gate
uv run python src/build_notebook.py && uv run python src/execute_notebook.py analysis.ipynb
uv run python src/figs_extra.py     # 추가 그림
uv run python src/screenshot_report.py report.html   # 시각 gate (8 PNG)
```

## 구조
- `src/` 수집·준비·노트북 빌드·품질·스크린샷 스크립트
- `docs/` 단계별 결정 로그 (문제→목표→소스→프로파일→방법→준비→결론)
- `data/` (gitignore) raw·cache·model_table — **알라딘 ToS: raw 텍스트 재배포 금지**, 집계만 공개
- `report.html` + `report_assets/` 최종 bilingual(ko/en) 리포트
- `analysis.ipynb` 분석 노트북

## 데이터 / 라이선스
도서 DB 제공 : **알라딘 인터넷서점(www.aladin.co.kr)**. TTB Open API, 비상업·분석 용도.
판매지수 = 최근가중 누적 판매 모멘텀 지수(≠매출). 관찰적 — 상관, 인과 아님.
