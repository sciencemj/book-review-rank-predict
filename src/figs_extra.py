#!/usr/bin/env python3
"""Two extra report figures: genre dominance + badge illusion decomposition."""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"AppleGothic","axes.unicode_minus":False,
                     "figure.dpi":120,"savefig.dpi":150,"savefig.bbox":"tight"})
ACC="#E8682B"; INK="#221C16"; MUT="#B0A48E"; SLATE="#5E6E8A"; CLAY="#C2613F"; GRN="#5E8A52"
ROOT="/Users/sciencemj/Desktop/data_science/Projects/book_review_rank_predict"
df=pd.read_csv(f"{ROOT}/data/model_table.csv"); G=json.load(open(f"{ROOT}/data/feature_groups.json"))
y=df["y"].values; T2=G["T2_badge"]; CTRL=G["control"]; CTX=G["context"]

# --- fig genre: mean log10 salesPoint by genre (the dominant axis) ---
gm=df.groupby("genre_grp")["y"].agg(["mean","count"]).sort_values("mean")
fig,ax=plt.subplots(figsize=(7,4.6))
ax.barh(gm.index, gm["mean"], color=[ACC if i==len(gm)-1 or i==0 else SLATE for i in range(len(gm))])
for i,(m,n) in enumerate(zip(gm["mean"],gm["count"])):
    ax.text(m+0.02,i,f"{m:.2f} (n={n})",va="center",fontsize=7.5,color=INK)
ax.set_xlabel("평균 log10(SalesPoint)"); ax.set_xlim(0,gm["mean"].max()*1.18)
ax.set_title("장르별 판매 모멘텀 — 장르 하나로 ~1.1 log(≈12배) 벌어짐")
ax.spines[["top","right"]].set_visible(False)
plt.savefig(f"{ROOT}/report/figures/05_genre.png"); plt.close()

# --- fig badge: the "2x" illusion vanishes under controls ---
def r2(Xdf): return sm.OLS(y, sm.add_constant(Xdf,has_constant="add")).fit().rsquared
def dm(cols):
    num=[c for c in cols if c!="genre_grp" and df[c].dtype!=object]; parts=[df[num].astype(float)]
    if "genre_grp" in cols: parts.append(pd.get_dummies(df["genre_grp"],prefix="g",drop_first=True).astype(float))
    return pd.concat(parts,axis=1)
g=dm(["genre_grp"]); ctx=dm(CTX); ctrlctx=pd.concat([dm(CTRL),ctx],axis=1)
raw=r2(df[T2].astype(float))
g_only=r2(pd.concat([g,df[T2].astype(float)],axis=1))-r2(g)
full=r2(pd.concat([ctrlctx,df[T2].astype(float)],axis=1))-r2(ctrlctx)
gap=df.groupby(df[["badge_bestseller","badge_award"]].max(axis=1).astype(bool))["y"].mean()
fig,ax=plt.subplots(figsize=(6.6,3.6))
labs=["raw\n(통제 전)","+장르만\n통제","+전체\n통제"]; vals=[raw,g_only,full]
bars=ax.bar(labs,vals,color=[CLAY,SLATE,GRN])
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.001,f"ΔR²={v:.3f}",ha="center",fontsize=9,color=INK)
ax.set_ylabel("badge 키워드 추가 설명력 ΔR²")
ax.set_title(f"'베스트셀러/수상 키워드 효과'(raw {10**(gap.iloc[1]-gap.iloc[0]):.1f}×)는\n장르를 통제하면 거의 사라진다 = 역인과/교란")
ax.spines[["top","right"]].set_visible(False); ax.set_ylim(0,raw*1.25)
plt.savefig(f"{ROOT}/report/figures/06_badge.png"); plt.close()
print("wrote 05_genre.png, 06_badge.png")
print(f"genre spread: {gm['mean'].min():.2f}..{gm['mean'].max():.2f}; badge raw {10**(gap.iloc[1]-gap.iloc[0]):.2f}x; ΔR² {raw:.3f}->{g_only:.3f}->{full:.3f}")
