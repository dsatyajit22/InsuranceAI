from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from .config import *

KEYWORDS={"water":"VS","flood":"VS","sewer":"VS","pipe":"VS","lightning":"VF","fire":"VF","smoke":"VF","vandal":"VE","stolen":"VE","theft":"VE","wind":"VE","glass":"VE","vehicle":"VE"}

class IntelligenceEngine:
    def __init__(self, claims):
        model_df=claims[claims.DescriptionClean.ne("")].copy()
        splitter=GroupShuffleSplit(1,test_size=TEST_SIZE,random_state=RANDOM_STATE)
        tr,te=next(splitter.split(model_df,groups=model_df.DescriptionClean))
        self.train=model_df.iloc[tr].reset_index(drop=True); self.test=model_df.iloc[te].reset_index(drop=True)
        self.classifier=Pipeline([("tfidf",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2)),("clf",LogisticRegression(max_iter=1000,class_weight="balanced"))])
        self.classifier.fit(self.train.DescriptionClean,self.train.target)
        self.retriever=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5)).fit(self.train.DescriptionClean)
        self.matrix=self.retriever.transform(self.train.DescriptionClean)
        amt=self.train.assign(logamt=np.log10(self.train.Claim.clip(lower=1))).groupby("target").logamt
        self.med=amt.median().to_dict(); self.mad=amt.apply(lambda x:(x-x.median()).abs().median()).replace(0,1e-6).to_dict()
    def classify(self,text):
        clean=''.join(ch.lower() for ch in str(text) if ch.isalnum())
        if not clean: raise ValueError("Enter a claim description.")
        p=self.classifier.predict_proba([clean])[0]; i=int(p.argmax())
        return clean,str(self.classifier.classes_[i]),float(p[i])
    def retrieve(self,clean,k,amount):
        sims=cosine_similarity(self.retriever.transform([clean]),self.matrix).ravel(); t=self.train.copy(); t["Similarity"]=sims
        cols=["claim_row_id","PolicyNum","Year","DescriptionRaw","target","Claim","county","Similarity"]
        m=t.sort_values("Similarity",ascending=False).drop_duplicates("DescriptionClean").head(max(3,min(10,int(k))))[cols]
        counts=Counter(m.target); recommendation,votes=counts.most_common(1)[0]
        median=float(m.Claim.median()); avg=float(m.Claim.mean()); pct=None if median==0 else (float(amount)-median)/median*100
        confidence=.7*(votes/len(m))+.3*float(m.Similarity.mean())
        return m,recommendation,confidence,median,avg,pct
    def assess(self,text,amount,recorded=None,k=5):
        clean,pred,confidence=self.classify(text); matches,retrieval,retrieval_conf,median,average,pct=self.retrieve(clean,k,amount)
        expected=next((code for word,code in KEYWORDS.items() if word in clean),None)
        mismatch=bool(recorded and expected and recorded!=expected); conflict=pred!=retrieval
        logamount=np.log10(max(float(amount),1)); z=abs((logamount-self.med.get(pred,logamount))/(1.4826*self.mad.get(pred,1e-6)))
        signals={"amount":min(z/6,1),"uncertainty":1-confidence,"mismatch":float(mismatch),"conflict":float(conflict)}
        contributions={key:round(100*RISK_WEIGHTS[key]*value,1) for key,value in signals.items()}; score=round(min(100,sum(contributions.values())),1)
        level="Low" if score<30 else "Medium" if score<60 else "High"; action={"Low":"PROCEED","Medium":"REVIEW","High":"HIGH-PRIORITY REVIEW"}[level]
        agreement="CONFLICT" if conflict else "STRONG" if min(confidence,retrieval_conf)>=.70 else "MEDIUM" if min(confidence,retrieval_conf)>=.55 else "WEAK"
        factors=[k.title() for k,v in contributions.items() if v>0]
        explanation=("; ".join(factors)+" require review.") if factors else "No material review signals detected."
        trace=[{"Step":"Intake","Status":"Complete","Detail":"Claim text and amount validated"},{"Step":"Classification","Status":"Complete","Detail":f"{pred} at {confidence:.1%} confidence"},{"Step":"Text-similarity retrieval","Status":"Complete","Detail":f"Retrieved {len(matches)} distinct historical descriptions"},{"Step":"Coverage validation","Status":"Flag" if mismatch else "Pass","Detail":f"Keyword expectation: {expected or 'not available'}"},{"Step":"Amount benchmark","Status":"Complete","Detail":f"Robust anomaly z-score: {z:.2f}"},{"Step":"Review routing","Status":action,"Detail":f"Priority score: {score:.1f}/100"}]
        return {"prediction":pred,"classifier_confidence":confidence,"retrieval_prediction":retrieval,"retrieval_confidence":retrieval_conf,"agreement":agreement,"recorded":recorded,"keyword_expected":expected,"mismatch":mismatch,"risk_score":score,"risk_level":level,"action":action,"contributions":contributions,"explanation":explanation,"peer_median":median,"peer_average":average,"pct_vs_median":pct,"matches":matches,"trace":trace}
