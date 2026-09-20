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


# In[ ]:


#AuditResult : Train/Test Split


# In[21]:


print("TRAIN DISTRIBUTION")
print(train["target"].value_counts())
print()

print("TEST DISTRIBUTION")
print(test["target"].value_counts())
print()

print("TRAIN %")
print((train["target"].value_counts(normalize=True)*100).round(2))
print()

print("TEST %")
print((test["target"].value_counts(normalize=True)*100).round(2))


# In[ ]:


#Segmentation Check


# In[22]:


# Random sample of segmented descriptions

sample = train[["Description", "desc_seg"]].sample(
    50,
    random_state=42
)

pd.set_option("display.max_colwidth", None)

sample


# In[23]:


# Misclassified examples

errors = test.copy()

errors["pred"] = pred

wrong = errors[errors["target"] != errors["pred"]]

print("Number of mistakes:", len(wrong))
print()

wrong[
    ["Description", "desc_seg", "target", "pred"]
].sample(
    min(50, len(wrong)),
    random_state=42
)


# In[24]:


from sklearn.metrics import confusion_matrix
import pandas as pd

labels = sorted(test["target"].unique())

cm = confusion_matrix(
    test["target"],
    pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{x}" for x in labels],
    columns=[f"Pred_{x}" for x in labels]
)

cm_df


# In[25]:


# Confidence audit

errors = test.copy()

errors["pred"] = pred

proba = baseline.predict_proba(test["Description"])

errors["confidence"] = proba.max(axis=1)

errors["correct"] = (
    errors["target"] == errors["pred"]
)

print(
    errors.groupby("correct")["confidence"]
    .describe()
)


# In[26]:


try:
    from sentence_transformers import SentenceTransformer
    print("INSTALLED")
except Exception as e:
    print("NOT INSTALLED")
    print(e)


# In[27]:


# Test the retrieval engine on a variety of claims

examples = [
    "lightningdamage",
    "waterdamage",
    "glassvandalism",
    "stormdamage",
    "theftoftools",
    "vehiclehitbuilding"
]

for q in examples:
    print("\n" + "="*80)
    print("QUERY:", q)
    print("="*80)

    display(
        similar_claims(q, k=5)[
            ["Description", "target", "similarity"]
        ]
    )


# In[28]:


def retrieve_claim_insights(text, k=5):

    matches = similar_claims(text, k=k)

    top_cov = matches["target"].mode()[0]

    coverage_conf = (
        matches["target"]
        .value_counts(normalize=True)
        .iloc[0]
        * 100
    )

    avg_claim = matches["Claim"].mean()
    med_claim = matches["Claim"].median()

    print("INPUT:", text)
    print()

    print("Recommended Coverage:", top_cov)
    print(f"Coverage Confidence: {coverage_conf:.1f}%")
    print()

    print(f"Average Similar Claim Amount: ${avg_claim:,.2f}")
    print(f"Median Similar Claim Amount: ${med_claim:,.2f}")
    print()

    return matches

retrieve_claim_insights("waterdamagetogymfloor")


# In[ ]:


#Building Accesor Explanation Engine


# In[29]:


from collections import Counter

def explain_retrieval(text, k=5):

    matches = similar_claims(text, k=k)

    top_cov = matches["target"].mode()[0]

    cov_counts = Counter(matches["target"])

    print("="*70)
    print("INPUT CLAIM")
    print("="*70)
    print(text)

    print("\nRECOMMENDED COVERAGE")
    print("="*70)
    print(top_cov)

    print("\nNEIGHBOR DISTRIBUTION")
    print("="*70)

    for cov, cnt in cov_counts.items():
        print(f"{cov}: {cnt}/{k}")

    print("\nREASONING")
    print("="*70)

    if "water" in text.lower():
        print("- Water-related language detected")
    if "fire" in text.lower():
        print("- Fire-related language detected")
    if "lightning" in text.lower():
        print("- Lightning-related language detected")
    if "glass" in text.lower():
        print("- Glass/vandalism-related language detected")
    if "theft" in text.lower():
        print("- Theft-related language detected")
    if "wind" in text.lower():
        print("- Wind/storm-related language detected")

    print("\nSIMILAR CLAIMS")
    print("="*70)

    display(
        matches[
            ["Description","target","Claim","similarity"]
        ]
    )

    return matches


# In[30]:


explain_retrieval("waterdamagetogymfloor")


# In[31]:


import numpy as np

def claim_peer_analysis(text, claim_amount, k=5):

    matches = similar_claims(text, k=k)

    avg_amt = matches["Claim"].mean()
    med_amt = matches["Claim"].median()

    pct_diff = (
        (claim_amount - med_amt)
        / med_amt
        * 100
    )

    print("="*70)
    print("CLAIM PEER ANALYSIS")
    print("="*70)

    print(f"Input Claim Amount: ${claim_amount:,.2f}")
    print(f"Median Similar Claim Amount: ${med_amt:,.2f}")
    print(f"Average Similar Claim Amount: ${avg_amt:,.2f}")

    print()

    print(
        f"Difference vs Median: "
        f"{pct_diff:+.1f}%"
    )

    if pct_diff > 100:
        print("⚠ Claim amount much higher than peers")

    elif pct_diff < -50:
        print("⚠ Claim amount much lower than peers")

    else:
        print("✓ Claim amount within expected range")

    print()

    return matches


# In[ ]:


#Claims Decision Support Engine


# In[32]:


claim_peer_analysis(
    "waterdamagetogymfloor",
    80000
)


# In[33]:


from collections import Counter

def coverage_recommendation(text, k=5):

    matches = similar_claims(text, k=k)

    counts = Counter(matches["target"])

    top_cov = counts.most_common(1)[0][0]

    confidence = (
        counts.most_common(1)[0][1]
        / k
        * 100
    )

    print("="*70)
    print("COVERAGE RECOMMENDATION")
    print("="*70)

    print("Recommended Coverage:", top_cov)
    print(f"Confidence: {confidence:.1f}%")

    print("\nDistribution")

    for cov, cnt in counts.items():
        print(f"{cov}: {cnt}")

    return matches


# In[ ]:


#Confidence-Based Coverage Recommendation


# In[34]:


coverage_recommendation("vehiclehitbuilding")


# In[35]:


from collections import Counter

def advanced_coverage_recommendation(text, k=5):

    matches = similar_claims(text, k=k)

    counts = Counter(matches["target"])

    top_cov = counts.most_common(1)[0][0]

    vote_conf = (
        counts.most_common(1)[0][1]
        / k
        * 100
    )

    similarity_strength = (
        matches["similarity"].mean()
        * 100
    )

    final_conf = (
        vote_conf * 0.7
        +
        similarity_strength * 0.3
    )

    print("="*70)
    print("ADVANCED COVERAGE RECOMMENDATION")
    print("="*70)

    print("Coverage:", top_cov)
    print(f"Vote Confidence: {vote_conf:.1f}%")
    print(f"Similarity Strength: {similarity_strength:.1f}%")
    print(f"Final Confidence: {final_conf:.1f}%")

    return matches


# In[36]:


advanced_coverage_recommendation(
    "vehiclehitbuilding"
)


# In[39]:


from collections import Counter

def interpret_confidence(score):

    if score >= 90:
        return "VERY HIGH"

    elif score >= 75:
        return "HIGH"

    elif score >= 60:
        return "MEDIUM"

    else:
        return "LOW"


def advanced_coverage_recommendation(text, k=5):

    matches = similar_claims(text, k=k)

    counts = Counter(matches["target"])

    top_cov = counts.most_common(1)[0][0]

    vote_conf = (
        counts.most_common(1)[0][1]
        / k
        * 100
    )

    similarity_strength = (
        matches["similarity"].mean()
        * 100
    )

    final_conf = (
        vote_conf * 0.7 +
        similarity_strength * 0.3
    )

    confidence_level = interpret_confidence(
        final_conf
    )

    print("="*70)
    print("ADVANCED COVERAGE RECOMMENDATION")
    print("="*70)

    print("Coverage:", top_cov)
    print(f"Vote Confidence: {vote_conf:.1f}%")
    print(f"Similarity Strength: {similarity_strength:.1f}%")
    print(f"Final Confidence: {final_conf:.1f}%")
    print(f"Recommendation Strength: {confidence_level}")

    return matches


# In[40]:


advanced_coverage_recommendation(
    "vehiclehitbuilding"
)


# In[41]:


from collections import Counter
import numpy as np

def interpret_confidence(score):

    if score >= 90:
        return "VERY HIGH"
    elif score >= 75:
        return "HIGH"
    elif score >= 60:
        return "MEDIUM"
    else:
        return "LOW"


def claim_intelligence(text, claim_amount, k=5):

    matches = similar_claims(text, k=k)

    # Coverage recommendation
    counts = Counter(matches["target"])

    top_cov = counts.most_common(1)[0][0]

    vote_conf = (
        counts.most_common(1)[0][1]
        / k
        * 100
    )

    similarity_strength = (
        matches["similarity"].mean()
        * 100
    )

    final_conf = (
        vote_conf * 0.7
        +
        similarity_strength * 0.3
    )

    confidence_level = interpret_confidence(
        final_conf
    )

    # Claim amount benchmarking

    avg_amt = matches["Claim"].mean()

    med_amt = matches["Claim"].median()

    pct_diff = (
        (claim_amount - med_amt)
        / med_amt
        * 100
    )

    print("="*80)
    print("CLAIM INTELLIGENCE REPORT")
    print("="*80)

    print(f"Input Description: {text}")
    print(f"Input Claim Amount: ${claim_amount:,.2f}")

    print("\nRECOMMENDED COVERAGE")
    print("-"*40)

    print(f"Coverage: {top_cov}")
    print(f"Confidence: {final_conf:.1f}%")
    print(f"Strength: {confidence_level}")

    print("\nHISTORICAL PEER ANALYSIS")
    print("-"*40)

    print(f"Median Claim Amount: ${med_amt:,.2f}")
    print(f"Average Claim Amount: ${avg_amt:,.2f}")

    print(
        f"Difference vs Median: "
        f"{pct_diff:+.1f}%"
    )

    if pct_diff > 100:
        print(
            "WARNING: Claim amount much higher than peers"
        )

    elif pct_diff < -50:
        print(
            "WARNING: Claim amount much lower than peers"
        )

    else:
        print(
            "Claim amount within expected range"
        )

    print("\nNEIGHBOR DISTRIBUTION")
    print("-"*40)

    for cov, cnt in counts.items():
        print(f"{cov}: {cnt}/{k}")

    print("\nTOP SIMILAR CLAIMS")
    print("-"*40)

    display(
        matches[
            [
                "Description",
                "target",
                "Claim",
                "similarity"
            ]
        ]
    )

    return matches


# In[42]:


claim_intelligence(
    text="vehiclehitbuilding",
    claim_amount=12000
)


# In[ ]:


#Deduped Retrieval func


# In[43]:


def similar_claims_dedup(text, k=5):

    q = retr_vec.transform([text])

    sims = cosine_similarity(
        q,
        TRAIN_MAT
    ).ravel()

    temp = train.copy()

    temp["similarity"] = sims

    temp = temp.sort_values(
        "similarity",
        ascending=False
    )

    # Remove duplicate descriptions

    temp = temp.drop_duplicates(
        subset=["Description"]
    )

    return temp[
        [
            "Description",
            "target",
            "Claim",
            "similarity"
        ]
    ].head(k)


# In[44]:


similar_claims_dedup(
    "vehiclehitbuilding",
    k=5
)


# In[45]:


def classifier_prediction(text):

    pred_cov = baseline.predict([text])[0]

    pred_proba = baseline.predict_proba([text])[0]

    clf_conf = pred_proba.max() * 100

    return pred_cov, clf_conf


# In[46]:


classifier_prediction(
    "vehiclehitbuilding"
)


# In[47]:


def agreement_engine(text, claim_amount):

    # ---------- CLASSIFIER ----------

    clf_cov = baseline.predict([text])[0]

    clf_proba = baseline.predict_proba([text])[0]

    clf_conf = clf_proba.max() * 100

    # ---------- RETRIEVAL ----------

    matches = similar_claims_dedup(text, k=5)

    counts = matches["target"].value_counts()

    retr_cov = counts.index[0]

    vote_conf = (
        counts.iloc[0] / 5 * 100
    )

    sim_strength = (
        matches["similarity"].mean() * 100
    )

    retr_conf = (
        vote_conf * 0.7 +
        sim_strength * 0.3
    )

    # ---------- AGREEMENT ----------

    agree = (
        clf_cov == retr_cov
    )

    if agree:

        final_cov = clf_cov

        final_conf = (
            clf_conf +
            retr_conf
        ) / 2

        status = "STRONG AGREEMENT"

    else:

        final_cov = "REVIEW REQUIRED"

        final_conf = min(
            clf_conf,
            retr_conf
        )

        status = "CONFLICT"

    # ---------- REPORT ----------

    print("="*80)
    print("CLASSIFIER ↔ RETRIEVAL AGREEMENT REPORT")
    print("="*80)

    print("\nCLASSIFIER")

    print(
        f"Coverage: {clf_cov}"
    )

    print(
        f"Confidence: {clf_conf:.1f}%"
    )

    print("\nRETRIEVAL")

    print(
        f"Coverage: {retr_cov}"
    )

    print(
        f"Confidence: {retr_conf:.1f}%"
    )

    print("\nAGREEMENT")

    print(
        f"Status: {status}"
    )

    print(
        f"Final Recommendation: {final_cov}"
    )

    print(
        f"Final Confidence: {final_conf:.1f}%"
    )

    return matches


# In[48]:


agreement_engine(
    text="vehiclehitbuilding",
    claim_amount=12000
)


# In[50]:


final_claim_intelligence(
    text="vehiclehitbuilding",
    claim_amount=12000
)


# In[51]:


from collections import Counter
import numpy as np

def final_claim_intelligence(text, claim_amount, k=5):

    # =====================================================
    # CLASSIFIER
    # =====================================================

    clf_cov = baseline.predict([text])[0]

    clf_proba = baseline.predict_proba([text])[0]

    clf_conf = clf_proba.max() * 100

    # =====================================================
    # RETRIEVAL
    # =====================================================

    matches = similar_claims_dedup(text, k=k)

    counts = Counter(matches["target"])

    retr_cov = counts.most_common(1)[0][0]

    vote_conf = (
        counts.most_common(1)[0][1]
        / k
        * 100
    )

    similarity_strength = (
        matches["similarity"].mean()
        * 100
    )

    retr_conf = (
        vote_conf * 0.7
        +
        similarity_strength * 0.3
    )

    # =====================================================
    # AGREEMENT ENGINE
    # =====================================================

    avg_conf = (
        clf_conf +
        retr_conf
    ) / 2

    if clf_cov == retr_cov:

        final_cov = clf_cov

        if clf_conf >= 70 and retr_conf >= 70:

            agreement = "STRONG"

            interpretation = (
                "Classifier and retrieval strongly support the same coverage."
            )

        elif clf_conf >= 55 and retr_conf >= 55:

            agreement = "MEDIUM"

            interpretation = (
                "Classifier and retrieval agree, but confidence is moderate."
            )

        else:

            agreement = "WEAK"

            interpretation = (
                "Classifier and retrieval agree, but confidence is low."
            )

        final_conf = avg_conf

    else:

        agreement = "CONFLICT"

        interpretation = (
            "Classifier and retrieval disagree. Human review recommended."
        )

        final_cov = "REVIEW REQUIRED"

        final_conf = min(
            clf_conf,
            retr_conf
        )

    # =====================================================
    # PEER ANALYSIS
    # =====================================================

    avg_amt = matches["Claim"].mean()

    med_amt = matches["Claim"].median()

    pct_diff = (
        (claim_amount - med_amt)
        / med_amt
        * 100
    )

    # =====================================================
    # REPORT
    # =====================================================

    print("="*90)
    print("FINAL CLAIM INTELLIGENCE REPORT")
    print("="*90)

    print(f"\nINPUT DESCRIPTION: {text}")
    print(f"INPUT CLAIM AMOUNT: ${claim_amount:,.2f}")

    print("\nCLASSIFIER")
    print("-"*40)

    print(f"Coverage: {clf_cov}")
    print(f"Confidence: {clf_conf:.1f}%")

    print("\nRETRIEVAL")
    print("-"*40)

    print(f"Coverage: {retr_cov}")
    print(f"Confidence: {retr_conf:.1f}%")

    print("\nAGREEMENT ANALYSIS")
    print("-"*40)

    print(f"Status: {agreement}")
    print(f"Interpretation: {interpretation}")

    print(f"\nFinal Recommendation: {final_cov}")
    print(f"Final Confidence: {final_conf:.1f}%")

    print("\nPEER BENCHMARK")
    print("-"*40)

    print(f"Median Similar Claim: ${med_amt:,.2f}")
    print(f"Average Similar Claim: ${avg_amt:,.2f}")

    print(
        f"Difference vs Median: {pct_diff:+.1f}%"
    )

    if pct_diff > 100:

        print(
            "⚠ HIGHER THAN HISTORICAL PEERS"
        )

    elif pct_diff < -50:

        print(
            "⚠ LOWER THAN HISTORICAL PEERS"
        )

    else:

        print(
            "✓ WITHIN EXPECTED RANGE"
        )

    print("\nNEIGHBOR DISTRIBUTION")
    print("-"*40)

    for cov, cnt in counts.items():

        print(f"{cov}: {cnt}/{k}")

    print("\nTOP SIMILAR CLAIMS")
    print("-"*40)

    display(
        matches[
            [
                "Description",
                "target",
                "Claim",
                "similarity"
            ]
        ]
    )

    return matches


# In[52]:


final_claim_intelligence(
    text="vehiclehitbuilding",
    claim_amount=12000
)


# In[ ]:




