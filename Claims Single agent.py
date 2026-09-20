#!/usr/bin/env python
# coding: utf-8

# # Single-Agent Orchestrator — Prototype (with local Ollama)
# 
# Turns the fixed claims pipeline into a genuine **single agent**: each stage is a **tool**, and one orchestrator **decides which tool to call next** based on the running state — skip OCR, call retrieval only when unsure, stop early on bad OCR, escalate high-risk claims to a human.
# 
# > Why it’s an *agent*, not a pipeline: control flow is **conditional on state** (OCR quality, classifier confidence, risk level), not a hardcoded 1→6 chain.
# 
# **Gen AI:** the explanation tool calls your **local Ollama model `llama3.2:1b`** (`localhost:11434`) — fully offline, no proxy issues — and falls back to a rule-based sentence if Ollama isn’t running. The agent’s *decisions* never depend on the LLM.

# In[1]:


# Single-Agent Orchestrator -- setup.
# Rebuilds the minimal pipeline pieces the agent needs as callable TOOLS.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import os

# Put CLAIMLEVEL.csv next to this notebook. (On a work laptop the URL download
# is often blocked by the proxy, so a local file is the reliable option.)
CLAIMLEVEL_LOCAL = "CLAIMLEVEL.csv"
CLAIMLEVEL_URL = "https://raw.githubusercontent.com/OpenActTexts/Loss-Data-Analytics/master/Data/CLAIMLEVEL.csv"
if os.path.exists(CLAIMLEVEL_LOCAL):
    claims = pd.read_csv(CLAIMLEVEL_LOCAL); print("Loaded local:", CLAIMLEVEL_LOCAL)
else:
    try:
        claims = pd.read_csv(CLAIMLEVEL_URL); print("Loaded from URL")
    except Exception as e:
        raise FileNotFoundError("Could not download CLAIMLEVEL.csv (work network likely blocked it). "
                                "Place the file next to this notebook or set CLAIMLEVEL_LOCAL to its full path.")

# minimal preprocess (same rules as the EDA/prototype notebooks)
vc = claims["CoverageCode"].value_counts(); keep = vc[vc >= 30].index.tolist()
claims["target"] = np.where(claims["CoverageCode"].isin(keep), claims["CoverageCode"], "Other")
claims["Fire5_clean"] = claims["Fire5"].where(claims["Fire5"].between(0, 10), np.nan)
claims["Fire5_clean"] = claims["Fire5_clean"].fillna(claims["Fire5_clean"].median())
claims = claims.drop_duplicates().reset_index(drop=True)
gss = GroupShuffleSplit(1, test_size=0.2, random_state=42)
tr, te = next(gss.split(claims, groups=claims["Description"]))
train, test = claims.iloc[tr].reset_index(drop=True), claims.iloc[te].reset_index(drop=True)
print("train/test:", train.shape, test.shape)


# In[2]:


# Fit the models the tools wrap (classifier, retriever, amount-anomaly stats)
clf = Pipeline([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
clf.fit(train["Description"], train["target"])
rvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(train["Description"])
RMAT = rvec.transform(train["Description"])
_g = train.assign(logamt=np.log10(train["Claim"].clip(lower=1))).groupby("target")["logamt"]
AMT_MED = _g.median().to_dict()
AMT_MAD = _g.apply(lambda x: (x - x.median()).abs().median()).replace(0, 1e-6).to_dict()
KEYWORD_COVERAGE = {"water": "VS", "flood": "VS", "sewer": "VS", "pipe": "VS",
                    "lightning": "VF", "fire": "VF", "smoke": "VF",
                    "vandal": "VE", "stolen": "VE", "theft": "VE", "wind": "VE", "glass": "VE"}
print("models ready")


# ## Tools — each pipeline stage the agent can call

# In[3]:


# ---------- TOOLS: each pipeline stage exposed as a callable the agent can choose ----------
def tool_ocr(image_text, quality=1.0):
    return {"text": image_text, "ocr_confidence": quality}

def tool_classify(text):
    proba = clf.predict_proba([text])[0]; i = int(proba.argmax())
    return {"prediction": clf.classes_[i], "confidence": float(proba[i])}

def tool_retrieve(text, k=5):
    sims = cosine_similarity(rvec.transform([text]), RMAT).ravel()
    seen, out = set(), []
    for j in sims.argsort()[::-1]:
        d = train.iloc[j]["Description"]
        if d in seen: continue
        seen.add(d); out.append((d, train.iloc[j]["target"], round(float(sims[j]), 3)))
        if len(out) >= k: break
    return {"similar": out}

def tool_validate(text, recorded):
    dl = str(text).lower()
    exp = next((v for k, v in KEYWORD_COVERAGE.items() if k in dl), None)
    return {"keyword_expected": exp, "mismatch": bool(exp is not None and exp != recorded)}

def tool_amount_anomaly(amount, coverage):
    la = np.log10(max(amount, 1)); med = AMT_MED.get(coverage, la); mad = AMT_MAD.get(coverage, 1e-6)
    return {"z_amount": round(float(abs((la - med) / (1.4826 * mad))), 2)}

def tool_explain_rules(state):
    "Deterministic fallback reason (used if the LLM is off/unreachable)."
    r = []
    if state.get("z_amount", 0) > 3: r.append(f"amount unusually high (z={state['z_amount']})")
    if state.get("mismatch"): r.append(f"text suggests {state['keyword_expected']} but coded {state.get('recorded')}")
    if state.get("confidence", 1) < 0.5: r.append(f"classifier unsure (conf={state['confidence']:.2f})")
    return "; ".join(r) if r else "no anomalies detected"

print("tools defined")


# ## Gen AI explanation via local Ollama

# In[4]:


# ---------- Gen AI explanation via LOCAL Ollama (llama3.2:1b) ----------
# Runs fully on your machine (http://localhost:11434) -> no internet/proxy issue,
# data never leaves the laptop. Falls back to the rule-based text if Ollama is off.
import json, urllib.request

OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_HOST  = "http://localhost:11434"
USE_LLM      = True     # set False to force the deterministic explanation

def _ollama_generate(prompt, model=OLLAMA_MODEL, host=OLLAMA_HOST, timeout=30):
    url = host.rstrip("/") + "/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    # bypass any corporate proxy for localhost, else it can hijack the call
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())["response"].strip()

def explain_with_llm(state):
    facts = tool_explain_rules(state)
    if not USE_LLM:
        return facts
    prompt = (
        "You are an insurance claim review assistant. In ONE short, plain sentence, tell the "
        "human reviewer why this claim needs attention. Use ONLY the signals given; do not invent facts.\n"
        f"Signals: {facts}.\n"
        f"Predicted coverage: {state.get('prediction')}, recorded coverage: {state.get('recorded')}, "
        f"risk score (0-100): {state.get('risk')}."
    )
    try:
        return _ollama_generate(prompt)
    except Exception as e:
        return facts + f"   [Ollama unavailable: {type(e).__name__}; used rule-based text]"

# quick connectivity check (won't crash if Ollama isn't running)
try:
    print("Ollama says:", _ollama_generate("Reply with the single word: ready"))
except Exception as e:
    print(f"Ollama not reachable ({type(e).__name__}). The agent will fall back to rule-based reasons.")
    print("To enable: install Ollama, run `ollama pull llama3.2:1b`, then `ollama serve`.")


# ## The orchestrator agent

# In[5]:


# ================= THE SINGLE ORCHESTRATOR AGENT =================
# An AGENT (not a fixed pipeline): at each step it DECIDES which tool to call
# next based on state -- skip OCR, call retrieval only when unsure, stop early
# on bad OCR, escalate to a human on high risk. The Gen AI explainer (Ollama)
# is used for the final reason only; the agent's decisions never depend on it.
def orchestrate(claim, conf_threshold=0.5, ocr_min=0.6, high_risk=40):
    trace, state = [], {}
    def log(step, decision, detail): trace.append((step, decision, detail))

    if claim.get("is_image"):                                   # DECISION 1 - image vs text
        o = tool_ocr(claim["text"], claim.get("ocr_quality", 1.0)); state.update(o)
        if o["ocr_confidence"] < ocr_min:                       # DECISION 2 - OCR quality gate
            log("intake/ocr", "ESCALATE", f"OCR {o['ocr_confidence']:.2f} < {ocr_min} -> request re-upload")
            return {"status": "ESCALATED_OCR", "trace": trace, "state": state}
        log("intake/ocr", "PROCEED", f"OCR ok ({o['ocr_confidence']:.2f})"); text = o["text"]
    else:
        log("intake", "PROCEED", "raw text -> skip OCR"); text = claim["text"]

    cl = tool_classify(text); state.update(cl); state["recorded"] = claim.get("recorded_coverage")
    if cl["confidence"] >= conf_threshold:                      # DECISION 3 - confidence branch
        log("classify", "CONFIDENT", f"{cl['prediction']} ({cl['confidence']:.2f}) -> skip retrieval"); need = False
    else:
        log("classify", "UNSURE", f"{cl['prediction']} ({cl['confidence']:.2f}) -> gather context"); need = True

    if need:                                                    # DECISION 4 - conditional retrieval
        state["similar"] = tool_retrieve(text)["similar"]
        log("retrieve", "CALLED", f"fetched {len(state['similar'])} distinct similar claims")

    state.update(tool_validate(text, claim.get("recorded_coverage")))
    state.update(tool_amount_anomaly(claim.get("amount", 0), cl["prediction"]))
    log("validate", "FLAG" if state["mismatch"] else "OK",
        (f"{state['keyword_expected']} vs {claim.get('recorded_coverage')}" if state["mismatch"] else "consistent"))

    risk = (0.5 * min(state.get("z_amount", 0) / 6, 1)          # DECISION 5 - route by risk
            + 0.3 * (1 if state["mismatch"] else 0)
            + 0.2 * (1 if cl["confidence"] < 0.5 else 0)) * 100
    state["risk"] = round(risk, 1)
    state["explanation"] = explain_with_llm(state)              # <-- Gen AI (Ollama) reason
    if risk >= high_risk:
        log("risk", "ESCALATE", f"risk {risk:.1f} >= {high_risk} -> human investigation"); status = "ESCALATED_HUMAN"
    else:
        log("risk", "AUTO", f"risk {risk:.1f} < {high_risk} -> auto-clear"); status = "AUTO_CLEARED"

    return {"status": status, "risk": state["risk"], "prediction": cl["prediction"],
            "confidence": round(cl["confidence"], 2), "explanation": state["explanation"], "trace": trace}
print("agent ready")


# In[6]:


def run(title, claim, **kw):
    res = orchestrate(claim, **kw)
    print("=" * 74); print(title); print("=" * 74)
    print(f"STATUS = {res['status']}   risk = {res.get('risk')}   "
          f"pred = {res.get('prediction')}   conf = {res.get('confidence')}")
    print("reason :", res.get("explanation"))
    print("--- agent decision trace ---")
    for step, decision, detail in res["trace"]:
        print(f"   [{step:12s}] {decision:10s} {detail}")
    print()
    return res


# ## Demonstration — five claims, five different decision paths

# In[7]:


# PATH A -- ordinary text claim: agent skips OCR & retrieval, auto-clears
_ = run("A) Normal text claim  (expect AUTO_CLEARED)",
        {"text": "lightningdamage", "recorded_coverage": "VF", "amount": 6800})


# In[8]:


# PATH B -- inflated amount: agent escalates to a human (LLM writes the reason)
_ = run("B) Inflated amount  (expect ESCALATED_HUMAN)",
        {"text": "lightningdamage", "recorded_coverage": "VF", "amount": 5_000_000})


# In[9]:


# PATH C -- text vs coverage mismatch: agent FLAGS possible mis-coding
_ = run("C) Coverage mismatch  (expect validate FLAG)",
        {"text": "waterdamageatcourthouse", "recorded_coverage": "VF", "amount": 9000})


# In[10]:


# PATH D -- image with poor OCR: agent STOPS EARLY, saves downstream compute
_ = run("D) Image, poor OCR  (expect ESCALATED_OCR, early stop)",
        {"text": "blurryscan", "is_image": True, "ocr_quality": 0.3,
         "recorded_coverage": "VE", "amount": 5000})


# In[11]:


# PATH E -- force the "unsure" branch to show CONDITIONAL retrieval firing
_ = run("E) Unsure classification  (expect retrieval CALLED)",
        {"text": "lightningdamage", "recorded_coverage": "VF", "amount": 6800},
        conf_threshold=0.99)


# ## Batch routing over the test set

# In[12]:


# ---------- Batch the agent over the test set: how does it route real claims? ----------
records = []
for _, r in test.iterrows():
    res = orchestrate({"text": r["Description"], "recorded_coverage": r["target"], "amount": r["Claim"]})
    records.append(res["status"])
routing = pd.Series(records).value_counts()
print(routing.to_string())


# In[13]:


pct = pd.Series(records).value_counts(normalize=True).mul(100).round(1)
print("Agent routing on the test set (%):"); print(pct.to_string())
print("\nThis is the 'Agent' deliverable: one orchestrator that conditionally")
print("calls tools, escalates only the claims that need a human, and uses the")
print("local Ollama model (llama3.2:1b) to write each plain-English reason.")


# In[ ]:




