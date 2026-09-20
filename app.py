import streamlit as st
from src.ui import page,metric_card
from src.state import datasets,engine
page("AI-Powered Insurance Claim Assessment","Executive analytics and explainable AI-assisted claim review using historical 2006–2010 data.")
raw_claims,raw_policy,claims,policy=datasets(); ai=engine()
st.markdown("### Enterprise decision-support workspace")
st.write("Explore portfolio performance, analyse historical claims, assess a new claim and inspect the deterministic agent workflow.")
a,b,c,d=st.columns(4)
with a: metric_card("Clean claim records",f"{len(claims):,}","#2563EB",f"{len(raw_claims)-len(claims):,} exact duplicates removed")
with b: metric_card("Policy-year records",f"{len(policy):,}","#0F766E","De-duplicated before analysis")
with c: metric_card("Coverage classes",f"{len(ai.classifier.classes_)}","#7C3AED","Rare classes grouped as Other")
with d: metric_card("Historical period","2006–2010","#F59E0B","Reference data, not current performance")
st.info("This application supports human review. It does not approve/reject claims and does not estimate fraud probability.")
