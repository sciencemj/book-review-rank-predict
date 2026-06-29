#!/usr/bin/env python3
"""Aladin TTB API corpus collector — Stage 3->4 bridge.

Genre-stratified, salesPoint-range-spanning sample of Korean books.
- Discovery: Bestseller per genre CategoryId, strided pages (high->low salesPoint).
- Filter: salesPoint > 0 AND age >= MIN_AGE_DAYS (index matured).
- Detail: one ItemLookUp per book -> itemPage (length) + ratingInfo (review counts) + packing.
- Resumable: every API response cached to data/cache/. Re-runs hit cache, not the API.
- Polite + sequential (shared 5,000/day quota; never parallelize).

Reads ALADIN_TTBKEY from .env. Never prints the key. Writes data/raw/corpus.csv.
"""
import json, urllib.request, urllib.parse, re, os, csv, time, sys
from datetime import date, datetime

ROOT = "/Users/sciencemj/Desktop/data_science/Projects/book_review_rank_predict"
CACHE = os.path.join(ROOT, "data", "cache")
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(CACHE, exist_ok=True); os.makedirs(RAW, exist_ok=True)
LOG = open(os.path.join(ROOT, "data", "collect.log"), "a")

def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg); LOG.write(msg + "\n"); LOG.flush()

with open(os.path.join(ROOT, ".env")) as f:
    KEY = next(l.split("=", 1)[1].strip() for l in f if l.startswith("ALADIN_TTBKEY"))
BASE = "http://www.aladin.co.kr/ttb/api/"

# ---- config ----
GENRE_SEEDS = [           # (CategoryId, seed label) — genre is re-derived from real categoryName
    (1, "소설/시/희곡"), (170, "경제경영"), (336, "자기계발"), (987, "과학"),
    (517, "예술/대중문화"), (1108, "어린이"), (1322, "외국어"), (1230, "가정/요리"),
    (55890, "건강/취미"), (74, "인문학"), (1196, "에세이"), (798, "사회과학"),
]
STRIDE_PAGES = [1, 2, 3, 5, 8, 12, 16, 20]   # rank 1..1000 gradient: high -> low salesPoint
MIN_AGE_DAYS = 90
MAX_BOOKS = 3200                              # cap on ItemLookUp calls
TODAY = date.today()
FETCH_TS = datetime.now().isoformat(timespec="seconds")
SLEEP = 0.12

if os.environ.get("SMOKE"):                  # tiny end-to-end test before the full run
    GENRE_SEEDS = GENRE_SEEDS[:2]
    STRIDE_PAGES = [1, 2]
    MAX_BOOKS = 40
    RAW = os.path.join(ROOT, "data", "raw_smoke"); os.makedirs(RAW, exist_ok=True)

_live_calls = 0
def call(ep, params, cache_key):
    """Cached GET. Returns parsed dict or {'_err':...}."""
    global _live_calls
    cf = os.path.join(CACHE, cache_key + ".json")
    if os.path.exists(cf):
        with open(cf) as fh:
            return json.load(fh)
    p = {**params, "ttbkey": KEY, "Version": "20131101", "Output": "js"}
    url = BASE + ep + "?" + urllib.parse.urlencode(p)
    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"}),
            timeout=30).read()
        txt = raw.decode("utf-8", "replace").strip()
        if txt.endswith(";"): txt = txt[:-1]
        try:
            data = json.loads(txt)
        except Exception:
            data = json.loads(re.sub(r"[\x00-\x1f]", " ", txt))
    except Exception as e:
        data = {"_err": f"{type(e).__name__}: {e}"}
    with open(cf, "w") as fh:
        json.dump(data, fh, ensure_ascii=False)
    _live_calls += 1
    time.sleep(SLEEP)
    return data

def genre_top(cat):
    parts = (cat or "").split(">")
    return parts[1] if len(parts) > 1 else (parts[0] if parts else "기타")

def age_days(pubdate):
    try:
        return (TODAY - datetime.strptime(pubdate, "%Y-%m-%d").date()).days
    except Exception:
        return None

# ---- 1) discovery ----
log(f"\n===== collect start {FETCH_TS} =====")
base_rows = {}   # isbn13 -> row dict
for cid, label in GENRE_SEEDS:
    seen_cid = set()
    for page in STRIDE_PAGES:
        r = call("ItemList.aspx",
                 {"QueryType": "Bestseller", "SearchTarget": "Book", "CategoryId": str(cid),
                  "MaxResults": "50", "start": str(page)},
                 cache_key=f"list_{cid}_{page}")
        items = r.get("item", []) if isinstance(r, dict) else []
        if not items:
            log(f"  cid {cid} p{page}: empty ({r.get('_err','')})"); continue
        isbns = [it.get("isbn13") for it in items if it.get("isbn13")]
        if isbns and set(isbns).issubset(seen_cid):
            log(f"  cid {cid} p{page}: WRAP detected, stop genre"); break
        seen_cid.update(isbns)
        for it in items:
            ib = it.get("isbn13")
            if not ib or ib in base_rows:
                continue
            base_rows[ib] = {
                "isbn13": ib, "title": it.get("title"), "author": it.get("author"),
                "publisher": it.get("publisher"), "salesPoint": it.get("salesPoint"),
                "description": it.get("description"), "categoryName": it.get("categoryName"),
                "genre_top": genre_top(it.get("categoryName")), "pubDate": it.get("pubDate"),
                "priceStandard": it.get("priceStandard"), "priceSales": it.get("priceSales"),
                "customerReviewRank": it.get("customerReviewRank"),
                "seed_cid": cid, "seed_genre": label,
            }
    log(f"  cid {cid} ({label}): cumulative unique={len(base_rows)}")

log(f"discovery done: {len(base_rows)} unique isbn13, live_calls={_live_calls}")

# ---- 2) filter: salesPoint>0 & matured ----
kept = []
for ib, row in base_rows.items():
    sp = row["salesPoint"]
    if not isinstance(sp, int) or sp <= 0:
        continue
    ad = age_days(row["pubDate"])
    if ad is None or ad < MIN_AGE_DAYS:
        continue
    row["age_days"] = ad
    kept.append(row)
log(f"after filter (sp>0 & age>={MIN_AGE_DAYS}d): {len(kept)}")

# ---- 3) stratified downsample to MAX_BOOKS by genre_top ----
if len(kept) > MAX_BOOKS:
    from collections import defaultdict
    by_g = defaultdict(list)
    for r in kept:
        by_g[r["genre_top"]].append(r)
    quota = max(1, MAX_BOOKS // len(by_g))
    sel = []
    for g, rows in by_g.items():
        rows.sort(key=lambda r: r["salesPoint"])          # keep range spread
        step = max(1, len(rows) // quota)
        sel.extend(rows[::step][:quota])
    kept = sel[:MAX_BOOKS]
    log(f"downsampled to {len(kept)} ({len(by_g)} genres, ~{quota}/genre)")

# ---- 4) detail lookups ----
fields = ["isbn13","title","author","publisher","salesPoint","description","categoryName",
          "genre_top","seed_genre","pubDate","age_days","priceStandard","priceSales",
          "customerReviewRank","itemPage","ratingScore","ratingCount","commentReviewCount",
          "myReviewCount","weight","sizeHeight","sizeWidth","sizeDepth","fetch_ts"]
out_path = os.path.join(RAW, "corpus.csv")
n = 0
with open(out_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for row in kept:
        ib = row["isbn13"]
        r = call("ItemLookUp.aspx",
                 {"ItemId": ib, "ItemIdType": "ISBN13", "OptResult": "ratingInfo,packing"},
                 cache_key=f"look_{ib}")
        it = (r.get("item") or [{}])[0] if isinstance(r, dict) else {}
        sub = it.get("subInfo") or {}
        ri = sub.get("ratingInfo") or {}
        pk = sub.get("packing") or {}
        row.update({
            "itemPage": sub.get("itemPage"),
            "ratingScore": ri.get("ratingScore"), "ratingCount": ri.get("ratingCount"),
            "commentReviewCount": ri.get("commentReviewCount"), "myReviewCount": ri.get("myReviewCount"),
            "weight": pk.get("weight"), "sizeHeight": pk.get("sizeHeight"),
            "sizeWidth": pk.get("sizeWidth"), "sizeDepth": pk.get("sizeDepth"),
            "fetch_ts": FETCH_TS,
        })
        w.writerow(row); n += 1
        if n % 100 == 0:
            fh.flush(); log(f"  lookup {n}/{len(kept)} (live_calls={_live_calls})")

log(f"DONE: wrote {n} rows -> {out_path}; total live_calls={_live_calls}")
LOG.close()
