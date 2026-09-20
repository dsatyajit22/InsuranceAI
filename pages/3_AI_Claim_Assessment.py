import streamlit as st
from src.ui import page,metric_card,risk_bar
from src.state import datasets,engine
from src.ocr import extract_document_text
page("AI Claim Assessment","Enter claim text or upload a document for OCR-assisted assessment.")
_,_,claims,policy=datasets(); ai=engine(); k=st.sidebar.slider("Similar historical claims",3,10,5)

def assess(text,amount,coverage,policy_num,source):
    try:
        st.session_state.v3_result=ai.assess(text,amount,coverage or None,k)
        st.session_state.v3_policy=policy_num; st.session_state.v3_source=source
    except ValueError as e: st.error(str(e))

t1,t2=st.tabs(["Enter Claim Text","Upload Claim Document"])
with t1:
    with st.form("text_claim"):
        left,right=st.columns([2,1]); text=left.text_area("Claim description",height=150)
        amount=right.number_input("Claim amount ($)",min_value=0.0,value=9000.0,step=500.0)
        cov=right.selectbox("Recorded coverage",["","VF","VS","VE","Other"]); pid=right.number_input("Policy number (optional)",min_value=0,value=0,step=1)
        go=st.form_submit_button("Assess Text Claim",type="primary",use_container_width=True)
    if go: assess(text,amount,cov,pid,"Text input")
with t2:
    st.caption("PNG, JPG, JPEG or PDF, maximum 10 MB")
    upload=st.file_uploader("Upload claim document",type=["png","jpg","jpeg","pdf"])
    if upload:
        result=extract_document_text(upload); st.success("Processing method: "+result["method"])
        if result["warning"]: st.warning(result["warning"])
        if upload.type.startswith("image/"): st.image(upload,width=420)
        with st.form("document_claim"):
            edited=st.text_area("Extracted text — review and correct",value=result["text"],height=180)
            a,b,c=st.columns(3); amount2=a.number_input("Claim amount ($)",min_value=0.0,value=9000.0,key="doc_amount")
            cov2=b.selectbox("Recorded coverage",["","VF","VS","VE","Other"],key="doc_cov"); pid2=c.number_input("Policy number",min_value=0,value=0,step=1,key="doc_pid")
            go2=st.form_submit_button("Assess Uploaded Claim",type="primary",use_container_width=True)
        if go2: assess(edited,amount2,cov2,pid2,result["method"])
if "v3_result" in st.session_state:
    r=st.session_state.v3_result; st.divider(); st.caption("Assessment source: "+st.session_state.get("v3_source","Text input"))
    cols=st.columns(5); vals=[("Predicted Coverage",r["prediction"],"#2563EB"),("Classifier Confidence",f"{r['classifier_confidence']:.1%}","#0F766E"),("Retrieval Recommendation",r["retrieval_prediction"],"#06B6D4"),("Agreement",r["agreement"],"#7C3AED"),("Recommended Action",r["action"],"#F59E0B")]
    for col,(label,value,color) in zip(cols,vals):
        with col: metric_card(label,value,color)
    st.plotly_chart(risk_bar(r["risk_score"]),use_container_width=True); st.info(r["explanation"])
    a,b,c=st.columns(3); a.metric("Peer median",f"${r['peer_median']:,.0f}"); b.metric("Peer average",f"${r['peer_average']:,.0f}"); c.metric("Difference vs median","N/A" if r["pct_vs_median"] is None else f"{r['pct_vs_median']:+.1f}%")
    table=r["matches"].copy(); table["Similarity"]=table.Similarity.map(lambda x:f"{x:.1%}"); table["Claim"]=table.Claim.map(lambda x:f"${x:,.0f}")
    st.subheader("Comparable historical claims"); st.dataframe(table.rename(columns={"DescriptionRaw":"Description","target":"Coverage"}),hide_index=True,use_container_width=True)
