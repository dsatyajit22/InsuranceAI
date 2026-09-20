import hashlib, re
import numpy as np
import pandas as pd
from .config import *

def _validate(df, required, name):
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"{name} missing columns: {', '.join(missing)}")

def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower()).strip()

def load_data():
    claims_raw=pd.read_csv(CLAIMS_PATH); policy_raw=pd.read_csv(POLICY_PATH)
    _validate(claims_raw,CLAIM_REQUIRED,"CLAIMLEVEL.csv"); _validate(policy_raw,POLICY_REQUIRED,"WiscPropFund.csv")
    # De-duplicate both datasets before dashboard calculations. WiscPropFund currently has no exact duplicates, but this makes the rule explicit.
    claims=claims_raw.drop_duplicates().reset_index(drop=True)
    policy=policy_raw.drop_duplicates().drop_duplicates(["PolicyNum","Year"]).reset_index(drop=True)
    claims["DescriptionRaw"]=claims["Description"].astype(str)
    claims["DescriptionClean"]=claims["DescriptionRaw"].map(normalize_text)
    claims["claim_row_id"]=[hashlib.sha1(f"{i}|{r.PolicyNum}|{r.Year}|{r.DescriptionRaw}|{r.Claim}".encode()).hexdigest()[:14] for i,r in claims.iterrows()]
    counts=claims["CoverageCode"].value_counts(); keep=counts[counts>=MIN_CLASS_SIZE].index
    claims["target"]=np.where(claims["CoverageCode"].isin(keep),claims["CoverageCode"],"Other")
    claims["CoverageLabel"]=claims["target"].map(COVERAGE_LABELS).fillna(claims["target"])
    policy["EntityLabel"]=policy["EntityType"].map(lambda x:f"Entity Type {int(x)}")
    return claims_raw, policy_raw, claims, policy

def filter_data(claims,policy,years,entities=None,counties=None,coverages=None):
    c=claims[claims.Year.between(*years)].copy(); p=policy[policy.Year.between(*years)].copy()
    if entities: c=c[c.EntityType.isin(entities)]
    if counties: c=c[c.county.isin(counties)]
    if coverages: c=c[c.target.isin(coverages)]
    return c,p
