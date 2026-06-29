# 도서 판매지수 분석 — 문제 & 목표 (Stage 1–2 확정)

## Stage 1 — 문제 제기 (확정)

**Problem statement:**
> 알라딘에서 수집한 여러 장르 도서 스냅샷에서, 출간경과일·리뷰수·평점(시간·인기 교란변수)을
> 통제한 상태에서 **책소개 멘트(마케팅 copy)의 텍스트 특성·장르·페이지수·가격**이
> 도서 판매지수(알라딘 SalesPoint)와 어떤 관계인지 규명하고, 이 요소들로 SalesPoint를 예측한다.

**핵심 reframe (정직 기록):**
1. **Target 교체:** Yes24 판매지수 → 알라딘 SalesPoint. Yes24는 합법 공개 API 없음.
   알라딘 TTB API가 같은 "최근가중 누적 판매지수" 계열 + 책소개·장르·페이지수·가격·평점 제공.
   결론은 "알라딘 기준"으로 명시한다(플랫폼 다름).
2. **Lever vs 통제변수:**
   - lever(의사결정 대상): 책소개 멘트(copy, 마케팅), 장르·페이지수(기획), 가격(잠재)
   - 통제변수(confounds): 출간경과일, 리뷰수, 평점 — 시간·인기 누적 효과
3. **인과 아님:** 관찰 데이터 → 상관/연관. "이렇게 쓰면 더 팔린다"는 인과 주장 불가. 한계로 명시.

**3-test:** 구체성 ✅ / 실재성 🟡(Stage 3 live 샘플로 확정) / 실행가능성 ✅(출판 copy·기획 가이드)

## Stage 2 — 분석 목표 (확정)

**한 문장 목표:** 위 problem statement.

**결정(decision):** 출판사·저자가 ① 책소개 copy 작성, ② 장르·분량·가격대 기획을 어떻게 할지.

**성공 기준 (2-part, 동등 합격 = 둘 다 충족해야 MET):**

- **비즈니스 합격선:**
  - ① 통제 후 SalesPoint와 통계적으로 유의한 lever 요인 **≥3개**를 효과 방향·크기와 함께 제시
    → copy/기획 가이드로 실제 활용 가능.
  - ② 신간 SalesPoint를 baseline보다 잘 추정해 사전 참고 가능.
- **정량 목표 (모델무관, baseline 상대, 구체 수치는 Stage 5에서 확정):**
  - 해석축: lever feature가 통제변수 **위에 추가 설명력** (incremental R² / partial F 유의)
    + 유의 요인 ≥3개 (다중검정 보정, 예: FDR).
  - 예측축: holdout/CV에서 log(SalesPoint) 예측이 **baseline 대비 개선** (R² 또는 MAE, 최소 ≥ baseline).
  - **둘 다 충족 = MET.**

**Baseline 정의:**
- floor baseline: 전체 평균 (또는 장르평균).
- control baseline: **출간경과일·리뷰수·평점만 쓴 모델.** lever(책소개/장르/길이/가격)의 순수 기여는
  이 control baseline 대비 개선으로 측정한다.

## 다음 (Stage 3)
알라딘 TTB API: TTBKey 발급 → endpoint/schema/필드(salesPoint·description·itemPage·category) live 샘플 →
라이선스(비상업·재배포·호출제한) 확인.
