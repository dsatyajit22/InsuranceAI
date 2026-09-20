#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis — Insurance Claim Assessment Capstone
# 
# EDA + preprocessing for the two project datasets. **Part A** = `CLAIMLEVEL.csv` (main / NLP data), **Part B** = `WiscPropFund.csv` (context / policy data), **Part C** = build the modelling-ready train/test split.
# 
# Run top to bottom. If you have the CSVs locally, put them next to this notebook; otherwise the loader reads them from the public GitHub repo.

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)


# In[2]:


# Load the datasets. The notebook first looks for a local copy; if not found,
# it reads the public file directly from the Loss-Data-Analytics GitHub repo.
import os

CLAIMLEVEL_LOCAL = "CLAIMLEVEL.csv"
WISCPROP_LOCAL   = "WiscPropFund.csv"

CLAIMLEVEL_URL = "https://raw.githubusercontent.com/OpenActTexts/Loss-Data-Analytics/master/Data/CLAIMLEVEL.csv"
WISCPROP_URL   = "https://raw.githubusercontent.com/OpenActTextDev/LDA_Ed2/main/Data/WiscPropFund.csv"

def load_csv(local_path, url):
    if os.path.exists(local_path):
        print(f"Loading local file: {local_path}")
        return pd.read_csv(local_path)
    print(f"Local file not found, reading from URL: {url}")
    return pd.read_csv(url)


# ---
# ## Part A — CLAIMLEVEL.csv (main dataset for NLP & classification)

# ### Shape, preview & structure

# In[3]:


claims = load_csv(CLAIMLEVEL_LOCAL, CLAIMLEVEL_URL)
claims.shape


# In[4]:


claims.head(10)


# In[5]:


claims.info()


# In[6]:


claims.isna().sum()


# In[7]:


claims.describe(include="number")


# In[8]:


cat_cols = ["ClaimStatus", "EntityType", "CoverageGroup", "CoverageCode", "county"]
claims[cat_cols].nunique()


# ### The `Description` text field (what the NLP model reads)

# In[9]:


# The Description field is the text our NLP model will read.
claims["desc_char_len"] = claims["Description"].astype(str).str.len()
claims["desc_word_count_raw"] = claims["Description"].astype(str).str.split().apply(len)
claims[["desc_char_len", "desc_word_count_raw"]].describe()


# In[10]:


# Sample of descriptions -- notice the spaces are stripped out
claims["Description"].dropna().sample(15, random_state=1).tolist()


# In[11]:


fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(claims["desc_char_len"], bins=30, color="#07A398", edgecolor="white")
ax[0].set_title("Description length (characters)")
ax[0].set_xlabel("characters")
ax[1].hist(claims["desc_word_count_raw"], bins=range(0, claims["desc_word_count_raw"].max()+2),
           color="#1C7293", edgecolor="white")
ax[1].set_title("Naive word count (before splitting)")
ax[1].set_xlabel("words")
plt.tight_layout()
plt.show()


# In[12]:


# Optional: preview word-segmentation. Uses 'wordninja' if installed.
try:
    import wordninja
    sample = claims["Description"].dropna().sample(8, random_state=2).tolist()
    for original in sample:
        split = " ".join(wordninja.split(original))
        print(f"{original:45s} -> {split}")
except ImportError:
    print("wordninja not installed. Install with: pip install wordninja")
    print("This step splits 'windblewstackoffanddamagedroof' -> 'wind blew stack off and damaged roof'.")


# ### Classification target candidates: `CoverageGroup` vs `CoverageCode`

# In[13]:


# Candidate classification target #1: CoverageGroup
claims["CoverageGroup"].value_counts(dropna=False)


# In[14]:


# Candidate classification target #2: CoverageCode
claims["CoverageCode"].value_counts(dropna=False)


# In[15]:


cov_group = claims["CoverageGroup"].value_counts(dropna=False)
cov_code  = claims["CoverageCode"].value_counts(dropna=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
cov_group.plot(kind="bar", ax=ax[0], color="#07A398", edgecolor="white")
ax[0].set_title("CoverageGroup distribution")
ax[0].set_ylabel("claims")
cov_code.plot(kind="bar", ax=ax[1], color="#1C7293", edgecolor="white")
ax[1].set_title("CoverageCode distribution")
ax[1].set_ylabel("claims")
plt.tight_layout()
plt.show()


# In[16]:


# Share of the single most common class -- shows how imbalanced the target is
for col in ["CoverageGroup", "CoverageCode"]:
    top_share = claims[col].value_counts(normalize=True).iloc[0]
    print(f"{col:15s} most-common class share: {top_share:.1%}  ({claims[col].nunique()} classes)")


# ### Claim status, entity type & the ClaimNum leakage check

# In[17]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
claims["ClaimStatus"].value_counts().plot(kind="bar", ax=ax[0], color="#12315E", edgecolor="white")
ax[0].set_title("ClaimStatus")
claims["EntityType"].value_counts().plot(kind="bar", ax=ax[1], color="#6D4C9A", edgecolor="white")
ax[1].set_title("EntityType")
plt.tight_layout()
plt.show()


# In[18]:


# Leakage check: how usable is ClaimNum as a grouping key?
n_rows = len(claims)
n_unique_claimnum = claims["ClaimNum"].nunique()
print(f"Total rows            : {n_rows}")
print(f"Unique ClaimNum values: {n_unique_claimnum}")
print(f"Rows that are a repeat of a ClaimNum: {n_rows - n_unique_claimnum}")


# In[19]:


# Most-repeated ClaimNum values -- if one value dominates, ClaimNum is NOT a
# reliable claim key and must not be used for grouping.
claims["ClaimNum"].value_counts().head(10)


# In[20]:


# Exact-duplicate (ClaimNum + Description) rows
dup_pairs = claims.duplicated(subset=["ClaimNum", "Description"]).sum()
print(f"Duplicate (ClaimNum, Description) rows: {dup_pairs}")


# ### Claim amount distribution

# In[21]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(claims["Claim"].clip(upper=claims["Claim"].quantile(0.99)), bins=40,
           color="#07A398", edgecolor="white")
ax[0].set_title("Claim amount (clipped at 99th pct)")
ax[0].set_xlabel("Claim ($)")
positive = claims["Claim"][claims["Claim"] > 0]
ax[1].hist(np.log10(positive), bins=40, color="#1C7293", edgecolor="white")
ax[1].set_title("Claim amount (log10)")
ax[1].set_xlabel("log10(Claim $)")
plt.tight_layout()
plt.show()


# In[22]:


claims["Claim"].describe()


# In[23]:


top_codes = claims["CoverageCode"].value_counts().head(6).index
subset = claims[claims["CoverageCode"].isin(top_codes)]
plt.figure(figsize=(10, 4))
sns.boxplot(data=subset, x="CoverageCode", y="Claim", hue="CoverageCode",
            showfliers=False, palette="crest", legend=False)
plt.yscale("log")
plt.title("Claim amount by CoverageCode (log scale, outliers hidden)")
plt.tight_layout()
plt.show()


# ### Time & geography

# In[24]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
claims["Year"].value_counts().sort_index().plot(kind="bar", ax=ax[0], color="#12315E", edgecolor="white")
ax[0].set_title("Claims per Year")
claims["county"].value_counts().head(12).plot(kind="barh", ax=ax[1], color="#0E7C66", edgecolor="white")
ax[1].set_title("Top 12 counties by claim count")
ax[1].invert_yaxis()
plt.tight_layout()
plt.show()


# In[25]:


num_cols = ["Year", "Claim", "Deduct", "Fire5"]
plt.figure(figsize=(6, 5))
sns.heatmap(claims[num_cols].corr(), annot=True, cmap="crest", fmt=".2f", square=True)
plt.title("Numeric correlation (CLAIMLEVEL)")
plt.tight_layout()
plt.show()


# ### Part A takeaways

# In[26]:


print("CLAIMLEVEL - key EDA takeaways")
print("-" * 55)
print(f"1. Rows: {len(claims):,} | Unique ClaimNum: {claims['ClaimNum'].nunique():,}"
      f" -> ClaimNum is NOT a usable key; dedupe by Description instead.")
print(f"2. Description median length: {int(claims['desc_char_len'].median())} chars,"
      f" spaces stripped -> add word-segmentation before NLP.")
grp_share = claims['CoverageGroup'].value_counts(normalize=True).iloc[0]
cod_share = claims['CoverageCode'].value_counts(normalize=True).iloc[0]
print(f"3. CoverageGroup is {grp_share:.0%} one class -> useless; use CoverageCode as target.")
print(f"4. CoverageCode top class = {cod_share:.0%}, long rare tail -> collapse rare classes"
      f" and report macro-F1.")
print(f"5. Claim amount highly skewed; Fire5 has junk values -> clean before modelling.")


# ---
# ## Part B — WiscPropFund.csv (context dataset — numeric only, no text)

# ### Shape, preview & structure

# In[27]:


policies = load_csv(WISCPROP_LOCAL, WISCPROP_URL)
policies.shape


# In[28]:


policies.head(10)


# In[29]:


policies.info()


# In[30]:


policies.isna().sum()


# In[31]:


# Confirm there is NO free-text column here
text_like = policies.select_dtypes(include="object").columns.tolist()
print("Text/object columns:", text_like if text_like else "NONE - all numeric")


# In[32]:


policies.describe()


# ### Numeric field distributions

# In[33]:


num_fields = ["Premium", "Deduct", "BCcov", "Freq"]
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
for a, col in zip(ax.ravel(), num_fields):
    data = policies[col]
    if col in ("Premium", "BCcov"):
        data = np.log10(data.replace(0, np.nan).dropna())
        a.set_xlabel(f"log10({col})")
    a.hist(data, bins=30, color="#07A398", edgecolor="white")
    a.set_title(col)
plt.tight_layout()
plt.show()


# In[34]:


freq_counts = policies["Freq"].value_counts().sort_index()
print(freq_counts)
plt.figure(figsize=(8, 4))
freq_counts.plot(kind="bar", color="#1C7293", edgecolor="white")
plt.title("Number of claims per policy-year (Freq)")
plt.xlabel("Freq"); plt.ylabel("policies")
plt.tight_layout()
plt.show()


# ### Indicator / categorical fields

# In[35]:


cat_like = ["Fire5", "NoClaimCredit", "EntityType", "AlarmCredit"]
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
for a, col in zip(ax.ravel(), cat_like):
    policies[col].value_counts().sort_index().plot(kind="bar", ax=a, color="#6D4C9A", edgecolor="white")
    a.set_title(col)
plt.tight_layout()
plt.show()


# In[36]:


zero_share = (policies["BCClaim"] == 0).mean()
print(f"Share of policy-years with zero BC claim amount: {zero_share:.1%}")
positive = policies["BCClaim"][policies["BCClaim"] > 0]
plt.figure(figsize=(8, 4))
plt.hist(np.log10(positive), bins=30, color="#0E7C66", edgecolor="white")
plt.title("BCClaim amount where > 0 (log10)")
plt.xlabel("log10(BCClaim $)")
plt.tight_layout()
plt.show()


# ### Correlations & Part B takeaways

# In[37]:


plt.figure(figsize=(8, 6))
sns.heatmap(policies.corr(numeric_only=True), annot=True, cmap="crest", fmt=".2f", square=False)
plt.title("Numeric correlation (WiscPropFund)")
plt.tight_layout()
plt.show()


# In[38]:


print("WiscPropFund - key EDA takeaways")
print("-" * 55)
print(f"1. Rows: {len(policies):,} (one row per policy-year).")
txt = policies.select_dtypes(include='object').columns.tolist()
print(f"2. Text columns: {txt if txt else 'NONE'} -> cannot support NLP.")
print(f"3. Freq and BCClaim are zero-inflated (most policies have no claim).")
print(f"4. Fields are premiums/coverage/deductibles -> background dashboard charts only.")
print(f"5. Keep separate from CLAIMLEVEL -- do not join row-by-row.")


# ---
# ## Part C — Preprocessing: build the modelling-ready dataset
# 
# Three data-driven fixes from the EDA above:
# 1. **Target** — collapse the rare `CoverageCode` classes into `Other` (the long tail is too small to learn).
# 2. **Clean** — drop single-valued columns (`ClaimStatus`, `CoverageGroup`) and fix `Fire5` sentinel/junk values.
# 3. **Leakage-safe split** — `ClaimNum` is not a usable key, so we dedupe on the row content and split by **grouping on `Description`** so identical text never appears in both train and test.

# In[39]:


# We work on a fresh copy of the claims table for modelling.
model_df = claims.copy()
print("Starting rows:", len(model_df))


# ### Fix 1 — define & collapse the classification target

# In[40]:


# --- Fix 1: define the target by collapsing rare CoverageCode classes ---
# EDA showed 3 classes (VE/VS/VF) cover ~99% of rows; the remaining classes
# have too few samples to train or evaluate. Collapse them into "Other".
TARGET_RAW      = "CoverageCode"
MIN_CLASS_SIZE  = 30          # classes smaller than this are merged into "Other"

class_counts = model_df[TARGET_RAW].value_counts()
keep_classes = class_counts[class_counts >= MIN_CLASS_SIZE].index.tolist()
model_df["target"] = np.where(model_df[TARGET_RAW].isin(keep_classes),
                              model_df[TARGET_RAW], "Other")

print("Kept classes (>= %d rows):" % MIN_CLASS_SIZE, keep_classes)
print(model_df["target"].value_counts())


# ### Fix 2 — drop no-signal columns & clean Fire5

# In[41]:


# --- Fix 2: drop no-signal columns and clean Fire5 ---
# ClaimStatus and CoverageGroup are single-valued (no signal). Fire5 has
# sentinel/junk values (e.g. -99, 90); keep only plausible 0..10 and impute.
drop_cols = [c for c in ["ClaimStatus", "CoverageGroup"] if model_df[c].nunique() <= 1]
model_df = model_df.drop(columns=drop_cols)

model_df["Fire5_clean"] = model_df["Fire5"].where(model_df["Fire5"].between(0, 10), np.nan)
median_fire5 = model_df["Fire5_clean"].median()
model_df["Fire5_clean"] = model_df["Fire5_clean"].fillna(median_fire5)

print("Dropped single-value columns:", drop_cols)
print("Fire5 range  before:", claims['Fire5'].min(), "to", claims['Fire5'].max())
print("Fire5 range  after :", model_df['Fire5_clean'].min(), "to", model_df['Fire5_clean'].max())


# ### Fix 3 — dedupe rows & leakage-safe grouped split

# In[42]:


# --- Fix 3a: remove exact-duplicate rows ---
# ClaimNum is NOT a usable key (one value dominates), so we dedupe on the
# actual record content instead of grouping by ClaimNum.
n_before = len(model_df)
model_df = model_df.drop_duplicates().reset_index(drop=True)
print(f"Exact-duplicate rows removed: {n_before - len(model_df)}  ->  {len(model_df)} rows remain")


# In[43]:


# --- Fix 3b: leakage-safe split by grouping on Description ---
# Identical claim descriptions must not appear in BOTH train and test, or the
# score is inflated. GroupShuffleSplit keeps all rows of a description together.
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(model_df, groups=model_df["Description"]))
train_df = model_df.iloc[train_idx].reset_index(drop=True)
test_df  = model_df.iloc[test_idx].reset_index(drop=True)

overlap = set(train_df["Description"]) & set(test_df["Description"])
print(f"Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")
print(f"Overlapping descriptions between train and test: {len(overlap)}  (must be 0)")


# In[44]:


# Check the target is spread proportionally across the split
dist = pd.DataFrame({
    "train": train_df["target"].value_counts(normalize=True).round(3),
    "test":  test_df["target"].value_counts(normalize=True).round(3),
})
dist


# ### Save the modelling-ready datasets

# In[45]:


# --- Save the modelling-ready datasets for the next stage ---
train_df.to_csv("claims_train.csv", index=False)
test_df.to_csv("claims_test.csv", index=False)
print("Saved: claims_train.csv, claims_test.csv")
print("\nPreprocessing complete. Person 1 can load these directly for modelling.")
print("Target column = 'target'  |  Text column = 'Description' (segment words next)")
print("Clean numeric feature available = 'Fire5_clean'")


# In[ ]:




