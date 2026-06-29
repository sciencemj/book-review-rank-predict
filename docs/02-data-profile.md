# Stage 4 — 데이터 이해 (real data 프로파일, 측정값)

수집: **1,814권** unique isbn13 (100% unique). fetch 2026-06-29. live calls 1,896 / 5,000.
discovery 4,318 → filter(salesPoint>0 & 출간 90일↑) 3,339 → **stratified downsample 1,814** (genre_top별 quota≈118, genre내 salesPoint range 고르게).

## Target: salesPoint
- raw: min 157, median 1,910, mean 3,694, max 67,292, **skew 4.61** (heavy right tail).
- **log10(salesPoint): skew 0.42 (near-normal), range 2.20–4.83.** → **모델 타깃 = log10(salesPoint).**
- mega-seller(>100k) 얇음 → 코퍼스는 mature backlist-leaning (한계).

## Genre
- 13개 주력 genre 각 **118권 (균형, 설계상)**: 소설/시/희곡·인문학·경제경영·외국어·과학·사회과학·자기계발·예술/대중문화·건강/취미·어린이·요리/살림·역사·여행.
- rare tail: 에세이95·대학교재48·청소년26·컴퓨터26·종교18·만화15·수험서14·유아12·좋은부모10·… 단발(잡지·고전·한국관련 1~5).
- **→ Stage 6: rare genre(<~30) "기타"로 collapse.** genre는 균형표본 = 인구비례 아님(한계, 리포트 명시).

## 결측·이상치
- null: **description 4.4% (80권 blurb 없음)**, itemPage/review/size ~0.1% (1~2권).
- **rating 0 = 무리뷰** (점수 0 아님): ratingCount==0 **16.8% (304권)**, 그중 303권 score==0. → ratingCount==0이면 rating=NaN + `has_reviews` flag.
- itemPage outlier: max 7,972(박스셋?), min 1 → cap/winsorize. price max 374k(전집).
- review counts / price / age 모두 skew → log1p.
- `fetch_ts` constant → 모델 drop (메타데이터).

## 교란변수 구조 (핵심 — Stage 5 설계 좌우)
log10(salesPoint) 상관:
| feature | pearson r | 메모 |
|---|---|---|
| log_reviews (rating+comment+my) | **+0.48** | **지배적 통제변수** |
| log_price (정가) | −0.14 | 비쌀수록 약간 낮음 |
| log_age (출간경과일) | **+0.03** | **거의 무관** — 지수가 recency-가중이라 나이≠판매. |
| log_page (페이지수) | +0.03 | raw 무상관 (통제 후 볼 것) |

- control baseline (log_age+log_reviews) **R²=0.233**.
- +price +page +genre dummies → **R²=0.513 (incremental +0.28)**.
- **→ lever feature가 변동의 ~77%를 두고 경쟁할 여지 있음. 관계·예측 둘 다 feasible.**

## 측정으로 굳힌 Stage 5 입력
- 타깃 변환 log10 필요(확정). 통제변수 = log_reviews(핵심), log_age(약하지만 포함), 평점(결측처리).
- lever = 책소개 텍스트 feature, genre(collapsed), log_page, log_price.
- 모델은 비선형·상호작용 가능(트리계열 후보) — raw 무상관(page)이 통제·상호작용 후 살아날 수 있음.
