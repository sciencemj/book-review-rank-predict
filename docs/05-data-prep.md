# Stage 6 — 데이터 준비 로그 (strict gate PASS, exit 0)

`src/prepare.py` → `data/model_table.csv` (1814행 38열, null 0). `data/feature_groups.json` = feature 그룹.

## 변환·결정 로그
- target `y = log10(salesPoint)` (salesPoint>0만; 이미 Stage4 필터됨).
- **텍스트 무impute:** `has_blurb`+`blurb_len`; null blurb은 길이 0, flag로 구분. 키워드는 `*_p100`(per-100자 rate)로 길이 decouple.
- **T1 copy (통제가능):** has_blurb, blurb_len, blurb_words, n_sentences, avg_sent_len, exclaim_p100, question_p100, has_quote, has_number_blurb.
- **T2 badge (placebo, 역인과):** badge_bestseller/rank/award/million/recommend/series_word + badge_count + badge_rate_p100. → 가이드 아님, 역인과 시연용.
- **product-type 유지(winsorize 금지):** is_set(세트|전N권|박스|>2000p), is_map(≤1p|지도). 세트·지도는 실존 product → flag로 보존.
- **context/control(actionable lever 아님):** genre_grp, log_page, log_price, is_set, is_map; log_reviews(=log1p rating+comment+my, **associational covariate=mediator**), log_ratingCount, has_reviews, ratingScore_imp(ratingCount==0→NaN→median impute), log_age.
- **OVB 통제(API 0, cache·corpus서):** is_series(seriesInfo 42.9%), author_freq_log(저자 표본내 빈도), pub_major(≥8권 출판사 flag), pub_freq.
- **rating 중복 제거:** customerReviewRank(≈ratingScore r=.998) drop. ratingScore_imp+has_reviews+log_ratingCount만.
- **genre 파서 가드:** 외국도서 처리; coarse 손매핑(유아→어린이, 대학교재·수험서·참고서→참고서/전문, 고전→인문, 좋은부모→가정/육아); <25 그룹 → 기타. 최종 18군(전부 ≥26).
- **work_id**(author_main + 정규화 title) → 1812 unique/1814. group_key=author_main → Stage7 GroupKFold/clustered SE.
- **leakage 차단:** priceSales/discount, customerReviewRank lever서 제외. reviews는 B2(사전런칭)서 제외 예정.
- constant 컬럼 자동 drop (fetch_ts/adult 등 미포함).

## gate
`data_quality.py --strict --key isbn13` → null≤0.40 PASS · full-row dup 0 · key dup 0 · constant 0 · **VERDICT PASS / exit 0**.
