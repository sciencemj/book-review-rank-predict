# Stage 5 — 방법 plan v2 (critic 3-lens 반영, v1 대체)

3 critic(statistical/business/data-quality)가 실데이터 재계산 → v1은 **false-success machine**:
genre(균형표집 artifact) + reviews(mediator)로 두 축 다 MET 뜨는데, 정작 결정 대상인 **copy 텍스트는 거의 무관**
(text marginal ΔR²≈0.009, genre 단독 +0.275). 강한 키워드 신호는 **역인과**(베스트셀러가 베스트셀러라 광고).

## 핵심 reframe (Stage 2 재정의)
표본 = "Aladin 장르 베스트셀러 리스트에 현재 올라있는 + salesPoint>0 + 출간 90일↑ 책"
= **결과변수로 선택된 표본**(D1). 모집단("모든 한국 책") 대표 아님 → 일반화 claim 금지.
Aladin엔 random-book 엔드포인트 없음(long tail=신간 salesPoint 0) → 재표집으로 못 고침 → **claim을 표본에 맞게 좁힘**.

**재정의 질문:** "현재 잘 팔리는(차트인) 한국 책들 사이에서, **책소개 copy**가 판매 모멘텀(log10 salesPoint)과
얼마나 관련 있나 — genre·누적인기(reviews)·저자 인지도·시리즈를 감안한 뒤 — 그리고 copywriter가 실제로 통제할 수 있는
copy 속성은 무엇인가?" → 출판사에 "blurb 문구에 얼마나 투자할 가치가 있나"를 정직하게 답함.

## 채택한 fix (critic 합의, 논쟁 없음)

### 성공기준 de-pool (false-success 차단)
- **lever block 통째 평가 금지.** 블록 분리, 각자 자기 bar:
  - **COPY-T1 블록** (copywriter 통제가능: blurb_len, 문장수, 평균문장길이, `!`/`?`수, 따옴표, 숫자유무, has_blurb) —
    *자체* ΔR² over [controls+genre+price+page+author/series/publisher] + *자체* FDR-유의 ≥3 (T1 중) + **효과크기 floor**(표준화β / salesPoint단위 효과 CI).
  - **T2 badge 블록** (베스트셀러·1위·수상·만부·노벨·완결·시리즈 등) → **placebo로 따로 보고**, 역인과 크기 시연용. **가이드 아님.**
  - genre·price·page → **context/control**, actionable lever로 계수 인용 금지 (genre 균형표집 artifact; price 도서정가제 10%캡·내생).
- **효과크기 게이트:** T1 효과를 salesPoint 단위로 역변환 + CI, 사전 명시한 실무적 크기 넘어야 "의미".

### 예측축 anchor 교체
- **B2 (사전런칭, reviews/rating 제외 = 진짜 실무모델)** vs **genre-평균 baseline** — 자체 threshold.
- B1(reviews 포함)은 **상한(upper bound)** 으로만 보고. (v1은 B1 threshold에 숨을 수 있었음)

### feature/treatment (data-quality lens)
- **description null(80, NOT MAR — 수험서 71% null·고령) 절대 impute 금지** → `has_blurb`+`blurb_len` flag, 키워드는 **per-100자 rate**, blurb_len·genre 항상 공변량.
- **product-type flag 유지(winsorize 금지):** `is_set`(세트|전\d+권|박스), `is_map/nonbook`(page==1·지도). 세트(전N권) 중앙 price 48k·1106p — 실존 신호. winsorize는 weight/size **오타**(height 1090mm 등)만.
- **OVB 통제 추가 (cache·corpus에 있음, API 0):** `is_series`(seriesInfo 43%), `author_freq`(저자 등장빈도 proxy), publisher major-flag/FE. → blurb 계수 상향편향 완화. (marketing비·표지·미디어는 미관측 → blurb 계수는 **상한**)
- **work_id**(정규화 title+author) + **GroupKFold by author/series** + author clustered SE (중복판·매거진 issue 누수 차단). isbn13 unique지만 work는 중복.
- **genre_top 파서 가드:** 외국도서 1행 처리; `기타` 블라인드 머지 대신 coarse 손매핑(참고서/수험서·유아 등). genre_top≠seed_genre 46% — 분석 genre는 categoryName 기준 명시.
- **rating 중복 제거:** customerReviewRank≈ratingScore (r=.998) → 하나만. keep `ratingScore`+`has_reviews`+`log_ratingCount`. score는 천장(median 9.5) 저정보.
- **log_reviews = associational 공변량(mediator/collider), "control" 호칭 안 함.** y=**현재 모멘텀**(flow), reviews=**누적 stock**.

## 정량 목표 (재확정, Stage 8 판정)
- **해석축(copy) MET:** COPY-T1 ΔR² ≥ 0.02 (full control+genre 모델 위) AND T1 중 FDR q<0.05 ≥3 AND 최소 1개 T1이 salesPoint ±10% 이상 효과(CI 하한 기준). *예상: 재계산상 copy marginal≈0.01 → NOT MET 가능성 — 정직한 결과로 수용.*
- **예측축 MET:** **B2** held-out test R²(log) ≥ genre-평균 baseline + 0.05 AND 의미있는 MAE 개선. B1은 상한 참고.
- 둘 다 + 효과크기 게이트 = MET. 미달이면 "copy는 모멘텀을 거의 못 움직임; genre·누적인기·저자/시리즈가 지배" = **강한 정직한 발견**.

## Stage 8/9 한계로 가져갈 것 (verbatim)
D1 결과변수-표집·range restriction; 단일 스냅샷(2026-06-29)·단일 리테일러(Aladin)·index≠매출/부수;
y=모멘텀 flow vs reviews=stock; backlist 생존자·hit tail 부재; 미관측 교란(marketing비·표지·미디어·각색) → 모든 copy 계수는 상한.
