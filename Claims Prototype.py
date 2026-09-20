#!/usr/bin/env python
# coding: utf-8

# # Claims Model — End-to-End Prototype
# 
# A working skeleton of the **whole claims pipeline** on `CLAIMLEVEL.csv`: preprocess → word-seg → baseline classifier → (optional) transformer → retrieval → anomaly risk score. Runs top-to-bottom. Each owner will deepen their own stage later.
# 
# _Prototype choices (swap later): TF-IDF char n-grams for the classifier & retrieval; a self-contained dictionary word-segmenter; DistilBERT and sentence-transformers are optional hooks._

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)


# In[2]:


# Load CLAIMLEVEL.csv (local copy if present, else public GitHub repo)
import os
CLAIMLEVEL_LOCAL = "CLAIMLEVEL.csv"
CLAIMLEVEL_URL = "https://raw.githubusercontent.com/OpenActTexts/Loss-Data-Analytics/master/Data/CLAIMLEVEL.csv"

if os.path.exists(CLAIMLEVEL_LOCAL):
    claims = pd.read_csv(CLAIMLEVEL_LOCAL)
    print("Loaded local:", CLAIMLEVEL_LOCAL)
else:
    claims = pd.read_csv(CLAIMLEVEL_URL)
    print("Loaded from URL")
print(claims.shape)
claims.head()


# ## 1 · Preprocess & leakage-safe split

# In[3]:


# Define target: collapse rare CoverageCode classes into "Other"
MIN_CLASS_SIZE = 30
counts = claims["CoverageCode"].value_counts()
keep_classes = counts[counts >= MIN_CLASS_SIZE].index.tolist()
claims["target"] = np.where(claims["CoverageCode"].isin(keep_classes),
                            claims["CoverageCode"], "Other")
print("Kept classes:", keep_classes)
claims["target"].value_counts()


# In[4]:


# Drop single-value columns and clean Fire5 junk values (-99, 90 ...)
drop_cols = [c for c in ["ClaimStatus", "CoverageGroup"] if claims[c].nunique() <= 1]
claims = claims.drop(columns=drop_cols)
claims["Fire5_clean"] = claims["Fire5"].where(claims["Fire5"].between(0, 10), np.nan)
claims["Fire5_clean"] = claims["Fire5_clean"].fillna(claims["Fire5_clean"].median())
print("Dropped:", drop_cols, "| Fire5 now:", claims["Fire5_clean"].min(), "to", claims["Fire5_clean"].max())


# In[5]:


# Remove exact-duplicate rows, then leakage-safe split by grouping on Description
claims = claims.drop_duplicates().reset_index(drop=True)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(claims, groups=claims["Description"]))
train = claims.iloc[tr_idx].reset_index(drop=True)
test  = claims.iloc[te_idx].reset_index(drop=True)
overlap = set(train["Description"]) & set(test["Description"])
print(f"train={train.shape} test={test.shape} | desc overlap={len(overlap)} (must be 0)")
print("NOTE: grouping by Description can leave a rare class entirely in train.")
print("      On the real 6k-row data this is minor; watch per-class support below.")


# In[6]:


# Word-segmentation: the descriptions have spaces stripped
# ("lightningdamageatwatertower"). Use wordninja if installed, else a small
# self-contained dictionary segmenter (good enough for a prototype).
LEXICON = set("""lightning damage water fire wind vandalism surge loss stolen equipment
surveillance copper wire tower glass school building roof stack blew burst pipe flooded
basement sewer backup leak ceiling siren airport park courthouse hail storm smoke theft
vehicle impact power electrical hydrant fence door at the to of and from misc""".split())

def _dict_segment(s):
    s2 = s.lower(); n = len(s2)
    best = [0] + [1e9] * n; back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - 15), i):
            w = s2[j:i]
            cost = best[j] + (1 if w in LEXICON else len(w) * 3 + 9)
            if cost < best[i]:
                best[i] = cost; back[i] = j
    out = []; i = n
    while i > 0:
        out.append(s2[back[i]:i]); i = back[i]
    return " ".join(reversed(out))

def segment(s):
    try:
        import wordninja
        return " ".join(wordninja.split(s))
    except Exception:
        return _dict_segment(str(s))

for ex in ["lightningdamageatwatertower", "windblewstackoffanddamagedroof", "waterdamageatcourthouse"]:
    print(f"{ex:35s} -> {segment(ex)}")


# ## 2 · Word-segmentation

# In[7]:


# Apply segmentation (kept as a separate column so raw text is preserved)
train["desc_seg"] = train["Description"].map(segment)
test["desc_seg"]  = test["Description"].map(segment)
train[["Description", "desc_seg"]].head()


# In[8]:


# Baseline classifier: TF-IDF (char n-grams) + Logistic Regression.
# Char n-grams work directly on the concatenated text, so this is robust even
# before word-segmentation. class_weight balances the imbalanced target.
baseline = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
baseline.fit(train["Description"], train["target"])
pred = baseline.predict(test["Description"])
print("macro-F1:", round(f1_score(test["target"], pred, average="macro"), 3))


# ## 3 · Baseline classifier (TF-IDF + Logistic Regression)

# In[9]:


# Per-class report -- watch recall on smaller classes (not just accuracy)
print(classification_report(test["target"], pred, zero_division=0))


# In[10]:


# Confusion matrix
labels = sorted(test["target"].unique())
cm = confusion_matrix(test["target"], pred, labels=labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="crest", xticklabels=labels, yticklabels=labels)
plt.xlabel("predicted"); plt.ylabel("actual"); plt.title("Baseline confusion matrix")
plt.tight_layout(); plt.show()


# In[11]:


# OPTIONAL transformer step (DistilBERT). Skips cleanly if transformers/torch
# are not installed. Person 1 runs this later to see if it beats the baseline.
try:
    from transformers import pipeline as hf_pipeline  # noqa
    import torch  # noqa
    print("transformers available -> a DistilBERT classifier can be trained here.")
    print("Placeholder: fine-tune distilbert-base-uncased on desc_seg -> target,")
    print("then compare macro-F1 against the baseline above and keep only if better.")
except Exception:
    print("transformers/torch NOT installed -> skipping DistilBERT for the prototype.")
    print("Baseline (TF-IDF + LogReg) stands as the working classifier.")
    print("To enable later: pip install transformers torch")


# ## 4 · Transformer (optional hook — DistilBERT)

# In[12]:


# Semantic retrieval: return the top-5 most similar past claims.
# Prototype uses TF-IDF cosine (swap for sentence-transformers later).
retr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(train["Description"])
TRAIN_MAT = retr_vec.transform(train["Description"])

def similar_claims(text, k=5):
    q = retr_vec.transform([text])
    sims = cosine_similarity(q, TRAIN_MAT).ravel()
    idx = sims.argsort()[::-1][:k]
    out = train.iloc[idx][["Description", "target", "Claim"]].copy()
    out["similarity"] = sims[idx].round(3)
    return out

similar_claims("lightningdamage")


# ## 5 · Similar-claim retrieval (top-5)

# In[13]:


# Risk scoring - Signal 1: amount anomaly.
# Robust z-score of log(Claim) within each target group (median/MAD).
risk = claims.copy()
risk["logamt"] = np.log10(risk["Claim"].clip(lower=1))
g = risk.groupby("target")["logamt"]
med = g.transform("median")
mad = g.transform(lambda x: (x - x.median()).abs().median()).replace(0, 1e-6)
risk["z_amount"] = ((risk["logamt"] - med) / (1.4826 * mad)).abs()
risk[["Description", "target", "Claim", "z_amount"]].sort_values("z_amount", ascending=False).head()


# ## 6 · Anomaly risk score (4 signals → 0–100)

# In[14]:


# Signal 2: duplicate / near-duplicate (exact Description + Claim match here)
risk["dup"] = risk.duplicated(subset=["Description", "Claim"], keep=False).astype(int)
print("Duplicate-flagged rows:", int(risk["dup"].sum()))


# In[15]:


# Signal 3: low classifier confidence (out-of-fold probabilities)
oof_proba = cross_val_predict(baseline, risk["Description"], risk["target"],
                              cv=5, method="predict_proba")
risk["confidence"] = oof_proba.max(axis=1)
risk["low_conf"] = (risk["confidence"] < 0.5).astype(int)
print("Low-confidence rows:", int(risk["low_conf"].sum()))


# In[16]:


# Signal 4: coverage-consistency (de-circularized).
# Out-of-fold prediction disagrees with recorded coverage AND an independent
# keyword rule agrees it is odd AND the model is confident -> possible mis-coding.
oof_pred = cross_val_predict(baseline, risk["Description"], risk["target"], cv=5)

KEYWORD_COVERAGE = {
    "water": "VS", "flood": "VS", "sewer": "VS", "pipe": "VS",
    "lightning": "VF", "fire": "VF", "smoke": "VF",
    "vandal": "VE", "stolen": "VE", "theft": "VE", "wind": "VE", "glass": "VE",
}
def keyword_expected(desc):
    d = str(desc).lower()
    for kw, cov in KEYWORD_COVERAGE.items():
        if kw in d:
            return cov
    return None

risk["kw_expected"] = risk["Description"].map(keyword_expected)
risk["mismatch"] = (
    (oof_pred != risk["target"]) &
    (risk["confidence"] > 0.6) &
    (risk["kw_expected"].notna()) &
    (risk["kw_expected"] != risk["target"])
).astype(int)
print("Consistency-flagged (possible mis-coding) rows:", int(risk["mismatch"].sum()))


# In[17]:


# Combine the four signals into a 0-100 risk score (weights are config).
def _norm(s):
    s = s.astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

W = {"z_amount": 0.40, "dup": 0.30, "low_conf": 0.20, "mismatch": 0.10}
risk["risk_score"] = (
    W["z_amount"] * _norm(risk["z_amount"]) +
    W["dup"]      * risk["dup"] +
    W["low_conf"] * risk["low_conf"] +
    W["mismatch"] * risk["mismatch"]
) * 100
risk["risk_score"].describe().round(1)


# In[18]:


# The most "worth a review" claims, with the reason columns visible
cols = ["Description", "target", "Claim", "z_amount", "dup", "low_conf", "mismatch", "risk_score"]
risk.sort_values("risk_score", ascending=False)[cols].head(10).round(2)


# ## Summary

# In[19]:


plt.figure(figsize=(7, 4))
plt.hist(risk["risk_score"], bins=40, color="#07A398", edgecolor="white")
plt.title("Distribution of risk scores (0-100)")
plt.xlabel("risk score"); plt.ylabel("claims")
plt.tight_layout(); plt.show()


# In[20]:


print("PROTOTYPE PIPELINE COMPLETE")
print("-" * 50)
print("Stages proven end-to-end on the claims dataset:")
print("  1. Preprocess + leakage-safe split")
print("  2. Word-segmentation")
print("  3. TF-IDF + LogReg baseline classifier (macro-F1)")
print("  4. DistilBERT hook (optional, run later)")
print("  5. Top-5 similar-claim retrieval")
print("  6. 4-signal anomaly risk score (0-100) with reasons")
print("\nThis is a skeleton. Each owner will deepen their stage.")

