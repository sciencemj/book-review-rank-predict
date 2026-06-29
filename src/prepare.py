#!/usr/bin/env python3
"""Stage 6 — build the modeling table with all critic-mandated fixes.

- T1 copy levers (copywriter-controllable) vs T2 badge levers (placebo, reverse-caused)
- text features never imputed: has_blurb + blurb_len; keyword features as rate per 100 chars
- product-type flags is_set / is_map (KEEP, don't winsorize)
- OVB controls: is_series (from cache), author_freq, publisher major-flag
- rating dup removed (customerReviewRank ~ ratingScore r=.998); ratingScore imputed + has_reviews
- log_reviews = associational covariate (mediator), y = current momentum (flow)
- work_id + author group key for GroupKFold / clustered SE
Writes data/model_table.csv + data/feature_groups.json.
"""
import pandas as pd, numpy as np, json, glob, os, re

ROOT = "/Users/sciencemj/Desktop/data_science/Projects/book_review_rank_predict"
df = pd.read_csv(f"{ROOT}/data/raw/corpus.csv")
print("loaded", len(df))

# ---- seriesInfo from cached lookups (no new API) ----
series_isbn = set()
for fp in glob.glob(f"{ROOT}/data/cache/look_*.json"):
    try:
        d = json.load(open(fp)); it = (d.get("item") or [{}])[0]
        if "seriesInfo" in it and it.get("seriesInfo"):
            ib = it.get("isbn13")
            if ib: series_isbn.add(int(ib))
    except Exception:
        pass
df["is_series"] = df["isbn13"].astype("int64").isin(series_isbn).astype(int)
print("is_series rate:", df["is_series"].mean().round(3))

# ---- target (flow: current momentum) ----
df = df[df["salesPoint"] > 0].copy()
df["y"] = np.log10(df["salesPoint"])

# ---- TEXT features (never impute; flag + length) ----
blurb = df["description"].fillna("")
df["has_blurb"] = (df["description"].notna() & (blurb.str.len() > 0)).astype(int)
df["blurb_len"] = blurb.str.len()
df["blurb_words"] = blurb.str.split().apply(len)
def n_sent(s):
    parts = re.split(r"[.!?…。！？]+", s)
    return max(1, sum(1 for p in parts if p.strip()))
df["n_sentences"] = blurb.apply(n_sent)
df["avg_sent_len"] = df["blurb_len"] / df["n_sentences"]
df["n_exclaim"] = blurb.str.count(r"[!！]")
df["n_question"] = blurb.str.count(r"[?？]")
# rates per 100 chars (decouple from length)
L = df["blurb_len"].clip(lower=1)
df["exclaim_p100"] = df["n_exclaim"] / L * 100
df["question_p100"] = df["n_question"] / L * 100
df["has_quote"] = blurb.str.contains(r"[\"'“”‘’『』「」«»]").astype(int)
df["has_number_blurb"] = blurb.str.contains(r"\d").astype(int)

# ---- T2 BADGE keywords (placebo: reverse-caused, NOT guidance) ----
BADGES = {
    "bestseller": ["베스트셀러","베스트 셀러","밀리언셀러","스테디셀러","베스트셀러 1위"],
    "rank":       ["1위","1 위","종합 1위","분야 1위","베스트 1위"],
    "award":      ["수상","노벨","부커","맨부커","퓰리처","문학상","대상 수상","수상작"],
    "million":    ["만 부","만부","십만","백만","밀리언","100만","누적 판매","누적 부수","돌파"],
    "recommend":  ["강력 추천","강력추천","화제작","화제의","필독","강추"],
    "series_word":["시리즈","완결","개정판","재출간","리커버","특별판","전권"],
}
def badge_flag(s, words): return int(any(w in s for w in words))
for name, words in BADGES.items():
    df[f"badge_{name}"] = blurb.apply(lambda s: badge_flag(s, words))
badge_cols = [f"badge_{k}" for k in BADGES]
df["badge_count"] = df[badge_cols].sum(axis=1)
df["badge_rate_p100"] = df["badge_count"] / L * 100

# ---- product-type flags (KEEP, don't winsorize) ----
title = df["title"].fillna("")
df["is_set"] = (title.str.contains(r"세트|전\s*\d+\s*권|박스|전권|컬렉션") |
                (df["itemPage"].fillna(0) > 2000)).astype(int)
df["is_map"] = ((df["itemPage"].fillna(99) <= 1) | title.str.contains(r"지도|전도|지명")).astype(int)

# ---- context / controls ----
df["log_page"] = np.log10(df["itemPage"].fillna(df["itemPage"].median()).clip(lower=1))
df["log_price"] = np.log10(df["priceStandard"].clip(lower=1000))
rev = df[["ratingCount","commentReviewCount","myReviewCount"]].fillna(0)
df["log_reviews"] = np.log1p(rev.sum(axis=1))           # associational covariate (mediator)
df["log_ratingCount"] = np.log1p(df["ratingCount"].fillna(0))
df["has_reviews"] = (df["ratingCount"].fillna(0) > 0).astype(int)
rs = df["ratingScore"].where(df["ratingCount"].fillna(0) > 0)   # 0 = no-review -> NaN
df["ratingScore_imp"] = rs.fillna(rs.median())
df["log_age"] = np.log10(df["age_days"].clip(lower=1))

# ---- OVB controls ----
df["author_main"] = df["author"].fillna("").apply(lambda s: re.split(r"[(,]", s)[0].strip())
af = df["author_main"].value_counts()
df["author_freq"] = df["author_main"].map(af)
df["author_freq_log"] = np.log1p(df["author_freq"])
pf = df["publisher"].fillna("기타").value_counts()
major_pubs = set(pf[pf >= 8].index)                     # prolific publishers
df["pub_major"] = df["publisher"].isin(major_pubs).astype(int)
df["pub_freq"] = df["publisher"].map(pf).fillna(1)

# ---- genre: guard parser + coarse map + merge rare ----
def gtop(cat):
    p = str(cat).split(">")
    if len(p) < 2: return "기타"
    if p[0] == "외국도서": return "외국도서"
    return p[1]
df["genre_top"] = df["categoryName"].apply(gtop)
COARSE = {"유아":"어린이","대학교재/전문서적":"참고서/전문","수험서/자격증":"참고서/전문",
          "초등학교참고서":"참고서/전문","중학교참고서":"참고서/전문","고등학교참고서":"참고서/전문",
          "고전":"인문학","좋은부모":"가정/육아","잡지":"기타","한국관련도서":"기타","외국도서":"기타"}
df["genre_grp"] = df["genre_top"].replace(COARSE)
gc = df["genre_grp"].value_counts()
rare = set(gc[gc < 25].index)
df["genre_grp"] = df["genre_grp"].apply(lambda g: "기타" if g in rare else g)
print("genre_grp:\n", df["genre_grp"].value_counts().to_string())

# ---- work_id (group key) ----
def norm_title(t):
    t = re.sub(r"\(.*?\)|\[.*?\]", "", str(t))
    t = re.sub(r"\s*(양장|특별판|개정판|전\s*\d+\s*권|세트|리커버).*", "", t)
    return re.sub(r"\s+", "", t).lower()[:30]
df["work_id"] = df["author_main"] + "::" + df["title"].apply(norm_title)

# ---- assemble + drop constant/dup cols ----
T1 = ["has_blurb","blurb_len","blurb_words","n_sentences","avg_sent_len",
      "exclaim_p100","question_p100","has_quote","has_number_blurb"]
T2 = badge_cols + ["badge_count","badge_rate_p100"]
CONTEXT = ["genre_grp","log_page","log_price","is_set","is_map"]
CONTROL = ["log_reviews","log_ratingCount","has_reviews","ratingScore_imp","log_age",
           "is_series","author_freq_log","pub_major","pub_freq"]
ID = ["isbn13","work_id","author_main","publisher","genre_top","salesPoint","y"]
keep = ID + T1 + T2 + CONTEXT + CONTROL
out = df[keep].copy()

# drop any accidental constant column (strict gate)
const_cols = [c for c in out.columns if out[c].nunique(dropna=False) <= 1]
if const_cols:
    print("dropping constant:", const_cols); out = out.drop(columns=const_cols)

out.to_csv(f"{ROOT}/data/model_table.csv", index=False)
groups = {"id": ID, "T1_copy": [c for c in T1 if c in out],
          "T2_badge": [c for c in T2 if c in out],
          "context": [c for c in CONTEXT if c in out],
          "control": [c for c in CONTROL if c in out],
          "target": "y", "group_key": "author_main"}
json.dump(groups, open(f"{ROOT}/data/feature_groups.json","w"), ensure_ascii=False, indent=2)
print(f"\nwrote model_table.csv ({len(out)} rows, {out.shape[1]} cols)")
print("n unique works:", df["work_id"].nunique(), "/ rows", len(df))
print("null counts (nonzero):\n", out.isna().sum()[out.isna().sum()>0].to_string() or "  none")
