# Stage 5 — 분석 방법 (plan, critic 검토 전 draft)

목표 2축(둘 다 합격): (A) 관계/해석 — 통제 후 유의 lever ≥3, (B) 예측 — baseline 대비 개선.
타깃 **y = log10(salesPoint)** (Stage4: skew 4.61→0.42).

## Feature engineering
**Levers (의사결정 대상):**
- 텍스트(책소개 blurb): 길이(char/word), 문장수, 평균 문장길이, `!`·`?` 수, 따옴표 유무,
  숫자/순위 표현 유무(예 "1위","10만부"), **마케팅 키워드 사전 flag/카운트**
  (베스트셀러·1위·화제·수상·추천·개정판·완결·시리즈·노벨·재출간 등), 저자/역자 다수 여부.
- genre (collapsed: rare<30 → 기타) — dummy.
- log_page (페이지수 lever), log_price (정가 lever).
- (예측축 보강 옵션) description TF-IDF 또는 ko 문장임베딩 — 손수 feature가 bar 못 넘으면 추가.

**Controls (교란):** log_reviews = log1p(ratingCount+commentReviewCount+myReviewCount) [핵심, r=.48],
log_age [약함 r=.03 but 포함], ratingScore (ratingCount==0이면 NaN→중앙값 impute + `has_reviews` flag).

## (A) 관계/해석
- **OLS** y ~ controls + levers, **robust SE (HC3)**. lever 계수 부호·크기·p.
- **다중검정 보정 (Benjamini-Hochberg FDR)** → 유의 lever ≥3 (q<0.05) = 합격선①.
- **Nested ΔR² + partial F**: (controls only) vs (controls+levers) → lever 추가 설명력 = 해석축 정량.
- 진단: VIF(다중공선성), 잔차·이분산(→robust SE), 영향점.

## (B) 예측
- **Baselines:** b0 전체평균, b1 genre-평균, **b2 control-only(reviews+age) [Stage2 지정 baseline]**.
- **Candidates:** Ridge/OLS(선형), **HistGradientBoosting**(비선형·상호작용; page/price raw≈0이나 조건부 효과 가능), RandomForest.
- **검증:** train/val/test 분할 + train서 5-fold CV 튜닝, **test는 최종 1회만** (튜닝에 test 미사용).
- **Metric:** test R²(log) + MAE(log) + 역변환 MAPE(salesPoint) 참고. metric별 baseline 대비 랭킹 → winner.
- permutation importance로 OLS 계수와 교차검증.

**누수 주의 + 2가지 예측 framing:**
- reviews는 출간후 결과 → "control"로 쓰면 lever 순효과 분리(해석엔 타당). 단 신간 사전예측엔 미존재.
- **(B1) 설명형 예측:** controls+levers vs b2 → lever ΔR² (Stage2 합격선 직접 대응).
- **(B2) 사전런칭 예측(실무 actionable):** reviews/rating 제외, copy+genre+page+price만 → vs genre-평균 baseline. 출판사가 출간 전 쓸 수 있는 모델. (보너스 actionability)

## 정량 목표 확정 (Stage 8 판정 기준)
- **해석축 MET:** lever block ΔR² ≥ 0.05 over controls AND partial F p<0.01 AND 유의 lever ≥3 (FDR q<0.05).
- **예측축 MET:** held-out test에서 (controls+levers) R²(log) ≥ control-baseline R² + **0.10** AND test R² ≥ **0.30**.
- **둘 다 충족 = MET.** (Stage4 in-sample: control 0.23, full 0.51 → held-out 하락 감안해도 도달 가능 추정.)

## 검증/한계 사전메모
- 관찰데이터 → 상관, 인과 아님. genre 균형표본(인구비례 아님). 텍스트=짧은 blurb.
- reviews=핵심 control이자 잠재 mediator(좋은 copy→리뷰↑→판매↑ 경로 차단 위험) → 해석 시 명시.
