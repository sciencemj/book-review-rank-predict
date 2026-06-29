# Stage 3 — 데이터 선정 (알라딘 TTB API, live 검증 완료)

## 접근
- TTBKey 무료, **5,000 calls/day** (basic). `.env`의 `ALADIN_TTBKEY` (코드에 하드코딩·출력 금지).
- Base `http://www.aladin.co.kr/ttb/api/{Endpoint}.aspx`, `Version=20131101`, `Output=js`(JSON 파싱 OK).
- 엔드포인트: `ItemList`(Bestseller/ItemNewAll…), `ItemSearch`(Keyword/Title/Author, Sort=SalesPoint/PublishTime…), `ItemLookUp`(isbn13별 상세).
- 페이지네이션: **50/page**, list 총 1000, search 총 다수(예: "소설" totalResults 17,073). MaxResults>50 줘도 50만 옴.

## 검증된 schema (live)
**base row (list/search에 이미 포함 — 추가 콜 불필요):**
| field | 의미 | 비고 |
|---|---|---|
| `isbn13` | join key | |
| `salesPoint` | **TARGET** 판매지수 | 매일 변함 → **fetch 시각 기록**. 베스트 22k~376k, 신간 0 |
| `description` | **책소개 blurb (텍스트 feature 원천)** | 가변 85~222자, 1~3문장. fullDescription은 API로 안 옴 |
| `categoryName` | 장르 (`>` 경로) | 예: `국내도서>어린이>게임 만화/캐릭터도감` |
| `pubDate` | 출간일 | 출간경과일(control) 계산 |
| `priceStandard`,`priceSales` | 정가/판매가 | lever(가격) |
| `customerReviewRank` | 평점 0~10 정수 | coarse |
| `publisher` | 출판사 | |

**ItemLookUp 추가분 (책별 1콜):**
| field | 의미 | 획득 |
|---|---|---|
| `subInfo.itemPage` | **페이지수 (lever, 책 길이)** | OptResult 없이 default로 나옴 ✓ |
| `subInfo.ratingInfo.{ratingScore,ratingCount,commentReviewCount,myReviewCount}` | 평점(소수)·리뷰수(control) | `OptResult=ratingInfo` |
| `subInfo.packing.{weight,sizeDepth/Height/Width}` | 물리 규격 | `OptResult=packing` (옵션) |

→ **책별 ItemLookUp 1콜** (`OptResult=ratingInfo,packing`)로 itemPage+리뷰수+규격 확보.

## 라이선스 (준수사항)
- 개인·비영리 분석 **허용**. 상업/기업 서비스 불가.
- **출처표기 필수(리포트에 박기):** `도서 DB 제공 : 알라딘 인터넷서점(www.aladin.co.kr)`
- **raw 책소개 텍스트 재배포 금지** → 계수·그래프·예측 등 **집계/파생 결과만 공개**. raw는 로컬 모델링용으로만.

## 표본 설계 (range restriction 방지)
- **장르 stratified** (소설/자기계발/경제경영/인문/과학/에세이/역사/어린이/예술/외국어 등) × **salesPoint 전 구간**.
- 수집 경로 혼합: Bestseller(고) + ItemSearch Sort=SalesPoint 깊은 페이지(중) + Sort=PublishTime/niche(저).
- **제외:** `salesPoint==0` (신간 미형성), `pubDate` 최근 < 90일 (지수 미성숙). isbn13 dedup.
- raw 응답 디스크 캐시(재실행 시 재호출 방지).
- target 코퍼스: 협의 후 확정 (~3–4천권 = 1일 한도 내).

## 한계 메모 (리포트 반영)
- 텍스트 = 짧은 blurb(1~3문장), 전체 책소개 아님.
- salesPoint = 알라딘 전용·매일 변동·시간가중 → 스냅샷, 인과 아님.
