#!/usr/bin/env python
# coding: utf-8

# # AI-Powered Insurance Claim Assessment System
# 

# In[14]:


import argparse
import json
import os
import random
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ## 0. Configuration
# 

# In[15]:


CLAIMLEVEL_LOCAL = "CLAIMLEVEL.csv"
CLAIMLEVEL_URL = (
    "https://raw.githubusercontent.com/OpenActTexts/"
    "Loss-Data-Analytics/master/Data/CLAIMLEVEL.csv"
)
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_HOST = "http://localhost:11434"


# ## 1. Data loading and model training
# 

# In[16]:


def load_claims():
    if os.path.exists(CLAIMLEVEL_LOCAL):
        claims = pd.read_csv(CLAIMLEVEL_LOCAL)
        print(f"Loaded local: {CLAIMLEVEL_LOCAL}")
    else:
        try:
            claims = pd.read_csv(CLAIMLEVEL_URL)
            print("Loaded from URL")
        except Exception:
            raise FileNotFoundError(
                "Could not find or download CLAIMLEVEL.csv. Place the file "
                "next to this script, or set CLAIMLEVEL_LOCAL to its path."
            )

    vc = claims["CoverageCode"].value_counts()
    keep = vc[vc >= 30].index.tolist()
    claims["target"] = np.where(
        claims["CoverageCode"].isin(keep), claims["CoverageCode"], "Other"
    )
    claims["Fire5_clean"] = claims["Fire5"].where(claims["Fire5"].between(0, 10), np.nan)
    claims["Fire5_clean"] = claims["Fire5_clean"].fillna(claims["Fire5_clean"].median())
    claims = claims.drop_duplicates().reset_index(drop=True)

    gss = GroupShuffleSplit(1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(claims, groups=claims["Description"]))
    train = claims.iloc[tr].reset_index(drop=True)
    test = claims.iloc[te].reset_index(drop=True)
    print("train/test:", train.shape, test.shape)
    return train, test


def fit_models(train):
    clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    clf.fit(train["Description"], train["target"])

    rvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(train["Description"])
    rmat = rvec.transform(train["Description"])

    g = train.assign(logamt=np.log10(train["Claim"].clip(lower=1))).groupby("target")["logamt"]
    amt_med = g.median().to_dict()
    amt_mad = g.apply(lambda x: (x - x.median()).abs().median()).replace(0, 1e-6).to_dict()

    keyword_coverage = {
        "water": "VS", "flood": "VS", "sewer": "VS", "pipe": "VS",
        "lightning": "VF", "fire": "VF", "smoke": "VF",
        "vandal": "VE", "stolen": "VE", "theft": "VE", "wind": "VE", "glass": "VE",
    }
    print("models ready")
    return clf, rvec, rmat, amt_med, amt_mad, keyword_coverage


# ## 2. Agent tools
# 

# In[17]:


class Tools:
    """Thin wrapper bundling every callable tool + the fitted models
    they wrap, so the agent can be instantiated once per (train, models)
    pair without relying on globals."""

    def __init__(self, train, clf, rvec, rmat, amt_med, amt_mad, keyword_coverage):
        self.train = train
        self.clf = clf
        self.rvec = rvec
        self.rmat = rmat
        self.amt_med = amt_med
        self.amt_mad = amt_mad
        self.keyword_coverage = keyword_coverage

    # ---- OCR ----------------------------------------------------------
    def tool_ocr(self, image_text, quality=1.0):
        """Simulated OCR read of a scanned claim form. In the real
        pipeline this calls the OCR module; here it takes the extracted
        text plus a confidence/quality score (0-1) supplied by that
        module (or synthesized for the demo, see run_ocr_demo)."""
        return {"text": image_text, "ocr_confidence": quality}

    # ---- Classification -------------------------------------------------
    def tool_classify(self, text):
        proba = self.clf.predict_proba([text])[0]
        i = int(proba.argmax())
        return {"prediction": self.clf.classes_[i], "confidence": float(proba[i])}

    # ---- Retrieval (similar past claims) --------------------------------
    def tool_retrieve(self, text, k=5):
        sims = cosine_similarity(self.rvec.transform([text]), self.rmat).ravel()
        seen, out = set(), []
        for j in sims.argsort()[::-1]:
            d = self.train.iloc[j]["Description"]
            if d in seen:
                continue
            seen.add(d)
            out.append((d, self.train.iloc[j]["target"], round(float(sims[j]), 3)))
            if len(out) >= k:
                break
        return {"similar": out}

    # ---- Consistency check ------------------------------------------
    def tool_validate(self, text, recorded):
        dl = str(text).lower()
        exp = next((v for k, v in self.keyword_coverage.items() if k in dl), None)
        return {"keyword_expected": exp, "mismatch": bool(exp is not None and exp != recorded)}

    # ---- Amount anomaly --------------------------------------------
    def tool_amount_anomaly(self, amount, coverage):
        la = np.log10(max(amount, 1))
        med = self.amt_med.get(coverage, la)
        mad = self.amt_mad.get(coverage, 1e-6)
        return {"z_amount": round(float(abs((la - med) / (1.4826 * mad))), 2)}

    # ---- Deterministic explanation (fallback if LLM is off) --------
    def tool_explain_rules(self, state):
        r = []
        if state.get("z_amount", 0) > 3:
            r.append(f"amount unusually high (z={state['z_amount']})")
        if state.get("mismatch"):
            r.append(f"text suggests {state['keyword_expected']} but coded {state.get('recorded')}")
        if state.get("confidence", 1) < 0.5:
            r.append(f"classifier unsure (conf={state['confidence']:.2f})")
        return "; ".join(r) if r else "no anomalies detected"


# ## 3. Local Ollama client
# 

# In[18]:


class OllamaClient:
    def __init__(self, model=OLLAMA_MODEL, host=OLLAMA_HOST, use_llm=True, timeout=30):
        self.model = model
        self.host = host
        self.use_llm = use_llm
        self.timeout = timeout
        self.available = self._probe()

    def _probe(self):
        if not self.use_llm:
            return False
        try:
            self.generate("Reply with the single word: ready")
            print(f"Ollama reachable -- using {self.model} for explanations.")
            return True
        except Exception as e:
            print(f"Ollama not reachable ({type(e).__name__}). "
                  "Falling back to rule-based text everywhere an LLM would be used.")
            print("To enable: install Ollama, run `ollama pull llama3.2:1b`, then `ollama serve`.")
            return False

    def generate(self, prompt, timeout=None):
        url = self.host.rstrip("/") + "/api/generate"
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode()
        # bypass any corporate proxy for localhost, else it can hijack the call
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode())["response"].strip()

    # -- (a) plain-English reason for a flagged/cleared claim -----------
    def explain(self, state, tools):
        facts = tools.tool_explain_rules(state)
        if not self.available:
            return facts
        prompt = (
            "You are an insurance claim review assistant. In ONE short, plain "
            "sentence, tell the human reviewer why this claim needs attention. "
            "Use ONLY the signals given; do not invent facts.\n"
            f"Signals: {facts}.\n"
            f"Predicted coverage: {state.get('prediction')}, recorded coverage: "
            f"{state.get('recorded')}, risk score (0-100): {state.get('risk')}."
        )
        try:
            return self.generate(prompt)
        except Exception as e:
            return facts + f"   [Ollama unavailable: {type(e).__name__}; used rule-based text]"

    # -- (b) generate a sample claim-form paragraph, for OCR demos ------
    FORM_FALLBACKS = {
        "fire": "Claimant reports a lightning strike caused fire damage to the "
                "roof and attic wiring of the property on the night of the storm. "
                "Estimated repair cost is six thousand eight hundred dollars.",
        "water": "Claimant reports a burst pipe in the basement caused water "
                  "damage to the flooring and drywall at the county courthouse. "
                  "Estimated repair cost is nine thousand dollars.",
        "theft": "Claimant reports a break-in overnight; a window was broken "
                  "and equipment was stolen from the storage room. Estimated "
                  "loss is five thousand dollars.",
    }

    def generate_claim_form(self, scenario="fire"):
        """Writes a short first-person claim description for a given
        scenario (fire/water/theft), to feed into the OCR demo. Falls
        back to a canned paragraph if Ollama is unavailable, so the OCR
        demo always has text to work with."""
        if not self.available:
            return self.FORM_FALLBACKS.get(scenario, self.FORM_FALLBACKS["fire"])
        prompt = (
            "Write a short (2-3 sentence) first-person insurance claim "
            f"description for a property {scenario} damage incident, in the "
            "plain style of a claim intake form. Do not add headers or labels, "
            "just the paragraph."
        )
        try:
            return self.generate(prompt)
        except Exception:
            return self.FORM_FALLBACKS.get(scenario, self.FORM_FALLBACKS["fire"])


# ## 4. Orchestrator agent
# 

# In[19]:


def orchestrate(claim, tools, llm, conf_threshold=0.5, ocr_min=0.6, high_risk=40):
    trace, state = [], {}

    def log(step, decision, detail):
        trace.append((step, decision, detail))

    # DECISION 1 -- image vs text (intake routing)
    if claim.get("is_image"):
        o = tools.tool_ocr(claim["text"], claim.get("ocr_quality", 1.0))
        state.update(o)
        # DECISION 2 -- OCR quality gate
        if o["ocr_confidence"] < ocr_min:
            log("intake/ocr", "ESCALATE",
                f"OCR {o['ocr_confidence']:.2f} < {ocr_min} -> request re-upload")
            return {"status": "ESCALATED_OCR", "trace": trace, "state": state}
        log("intake/ocr", "PROCEED", f"OCR ok ({o['ocr_confidence']:.2f})")
        text = o["text"]
    else:
        log("intake", "PROCEED", "raw text -> skip OCR")
        text = claim["text"]

    cl = tools.tool_classify(text)
    state.update(cl)
    state["recorded"] = claim.get("recorded_coverage")

    # DECISION 3 -- confidence branch
    if cl["confidence"] >= conf_threshold:
        log("classify", "CONFIDENT", f"{cl['prediction']} ({cl['confidence']:.2f}) -> skip retrieval")
        need_retrieval = False
    else:
        log("classify", "UNSURE", f"{cl['prediction']} ({cl['confidence']:.2f}) -> gather context")
        need_retrieval = True

    # DECISION 4 -- conditional retrieval
    if need_retrieval:
        state["similar"] = tools.tool_retrieve(text)["similar"]
        log("retrieve", "CALLED", f"fetched {len(state['similar'])} distinct similar claims")

    state.update(tools.tool_validate(text, claim.get("recorded_coverage")))
    state.update(tools.tool_amount_anomaly(claim.get("amount", 0), cl["prediction"]))
    log("validate", "FLAG" if state["mismatch"] else "OK",
        (f"{state['keyword_expected']} vs {claim.get('recorded_coverage')}"
         if state["mismatch"] else "consistent"))

    # DECISION 5 -- blend signals into a risk score & route
    risk = (
        0.5 * min(state.get("z_amount", 0) / 6, 1)
        + 0.3 * (1 if state["mismatch"] else 0)
        + 0.2 * (1 if cl["confidence"] < 0.5 else 0)
    ) * 100
    state["risk"] = round(risk, 1)
    state["explanation"] = llm.explain(state, tools)  # <-- Gen AI (Ollama) reason

    if risk >= high_risk:
        log("risk", "ESCALATE", f"risk {risk:.1f} >= {high_risk} -> human investigation")
        status = "ESCALATED_HUMAN"
    else:
        log("risk", "AUTO", f"risk {risk:.1f} < {high_risk} -> auto-clear")
        status = "AUTO_CLEARED"

    return {
        "status": status,
        "risk": state["risk"],
        "prediction": cl["prediction"],
        "confidence": round(cl["confidence"], 2),
        "explanation": state["explanation"],
        "trace": trace,
    }


def run(title, claim, tools, llm, **kw):
    res = orchestrate(claim, tools, llm, **kw)
    print("=" * 74)
    print(title)
    print("=" * 74)
    print(f"STATUS = {res['status']}   risk = {res.get('risk')}   "
          f"pred = {res.get('prediction')}   conf = {res.get('confidence')}")
    print("reason :", res.get("explanation"))
    print("--- agent decision trace ---")
    for step, decision, detail in res["trace"]:
        print(f"   [{step:12s}] {decision:10s} {detail}")
    print()
    return res


# ## 5. OCR demo helpers
# 

# In[20]:


def simulate_scan_noise(text, quality):
    """Corrupts a clean claim-form paragraph to mimic a low-quality
    scan, roughly in proportion to (1 - quality). This is a stand-in
    for the real OCR module's own confidence estimate."""
    if quality >= 0.95:
        return text
    rng = random.Random(42)
    chars = list(text)
    n_corrupt = int(len(chars) * (1 - quality) * 0.5)
    for _ in range(n_corrupt):
        idx = rng.randrange(len(chars))
        chars[idx] = rng.choice("#@$%?* ")
    return "".join(chars)


def run_ocr_demo(tools, llm):
    print("#" * 74)
    print("# OCR DEMO FORMS -- LLM-generated claim text at 3 scan qualities")
    print("#" * 74)
    scenarios = [
        ("fire", 0.95, "VF", 6800),   # clean scan -> should PROCEED
        ("water", 0.75, "VF", 9000),  # medium scan, above ocr_min -> PROCEED (may mismatch coverage)
        ("theft", 0.30, "VE", 5000),  # poor scan, below ocr_min -> ESCALATED_OCR
    ]
    for scenario, quality, recorded_coverage, amount in scenarios:
        clean_text = llm.generate_claim_form(scenario)
        scanned_text = simulate_scan_noise(clean_text, quality)
        print(f"\n--- scenario: {scenario} | simulated scan quality: {quality:.2f} ---")
        print("clean form text   :", clean_text)
        print("simulated OCR read:", scanned_text)
        run(
            f"OCR demo -- {scenario} claim (quality={quality:.2f})",
            {"text": scanned_text, "is_image": True, "ocr_quality": quality,
             "recorded_coverage": recorded_coverage, "amount": amount},
            tools, llm,
        )


# ## 6. Demonstration paths
# 

# In[21]:


def run_demo_paths(tools, llm):
    run("A) Normal text claim  (expect AUTO_CLEARED)",
        {"text": "lightningdamage", "recorded_coverage": "VF", "amount": 6800},
        tools, llm)

    run("B) Inflated amount  (expect ESCALATED_HUMAN)",
        {"text": "lightningdamage", "recorded_coverage": "VF", "amount": 5_000_000},
        tools, llm)

    run("C) Coverage mismatch  (expect validate FLAG)",
        {"text": "waterdamageatcourthouse", "recorded_coverage": "VF", "amount": 9000},
        tools, llm)

    run("D) Image, poor OCR  (expect ESCALATED_OCR, early stop)",
        {"text": "blurryscan", "is_image": True, "ocr_quality": 0.3,
         "recorded_coverage": "VE", "amount": 5000},
        tools, llm)

    run("E) Unsure classification  (expect retrieval CALLED)",
        {"text": "lightningdamage", "recorded_coverage": "VF", "amount": 6800},
        tools, llm, conf_threshold=0.99)


# ## 7. Batch routing
# 

# In[22]:


def run_batch_routing(test, tools, llm):
    print("#" * 74)
    print("# BATCH ROUTING OVER THE TEST SET")
    print("#" * 74)
    records = []
    for _, r in test.iterrows():
        res = orchestrate(
            {"text": r["Description"], "recorded_coverage": r["target"], "amount": r["Claim"]},
            tools, llm,
        )
        records.append(res["status"])
    routing = pd.Series(records).value_counts()
    pct = pd.Series(records).value_counts(normalize=True).mul(100).round(1)
    print(routing.to_string())
    print("\nAgent routing on the test set (%):")
    print(pct.to_string())
    print(
        "\nThis is the 'Agent' deliverable: one orchestrator that conditionally "
        "calls tools, escalates only the claims that need a human, and uses the "
        "local Ollama model (llama3.2:1b) to write each plain-English reason."
    )


# ## 8. Initialize data, models, tools, and Ollama
# Make sure `CLAIMLEVEL.csv` is in the same folder as this notebook if the URL cannot be reached.
# 

# In[11]:


train, test = load_claims()
clf, rvec, rmat, amt_med, amt_mad, keyword_coverage = fit_models(train)
tools = Tools(train, clf, rvec, rmat, amt_med, amt_mad, keyword_coverage)
llm = OllamaClient(use_llm=True)


# ## 9. Run demos
# 

# In[12]:


run_demo_paths(tools, llm)


# In[13]:


run_ocr_demo(tools, llm)


# ## 10. Optional: run batch routing
# 

# In[23]:


run_batch_routing(test, tools, llm)

