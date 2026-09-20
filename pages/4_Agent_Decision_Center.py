import streamlit as st
from src.ui import page,metric_card
page("Agent Decision Center","Transparent step-by-step orchestration of classification, retrieval, validation and review routing.")
if 'v3_result' not in st.session_state:
    st.info('Run an assessment from AI Claim Assessment first.')
else:
    r=st.session_state.v3_result
    a,b,c=st.columns(3)
    with a: metric_card('Final Action',r['action'],'#F59E0B')
    with b: metric_card('Review Priority',f"{r['risk_score']:.1f} / 100",'#DC2626')
    with c: metric_card('Evidence Agreement',r['agreement'],'#7C3AED')
    st.markdown('### Decision journey')
    for i,step in enumerate(r['trace'],1):
        status=step['Status']; css='ok' if status in ['Complete','Pass','PROCEED'] else 'warn'
        st.markdown(f"<div class='timeline {css}'><div class='step'>{i}</div><div><b>{step['Step']}</b><span>{status}</span><p>{step['Detail']}</p></div></div>",unsafe_allow_html=True)
    st.markdown('### Review-priority contribution')
    st.bar_chart(r['contributions'],horizontal=True)
    st.caption('The workflow provides decision support. A human reviewer remains responsible for the final claim decision.')
