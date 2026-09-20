import streamlit as st
import plotly.express as px
from src.ui import page,metric_card,style_fig
from src.state import datasets
from src.data import filter_data
page("Executive Overview","A concise portfolio view of premium, exposure, historical loss and claim activity.")
_,_,claims,policy=datasets(); years=st.sidebar.slider("Year range",int(policy.Year.min()),int(policy.Year.max()),(int(policy.Year.min()),int(policy.Year.max())))
entities=st.sidebar.multiselect("Claim entity type",sorted(claims.EntityType.unique()),default=[]); counties=st.sidebar.multiselect("County",sorted(claims.county.unique()),default=[]); coverages=st.sidebar.multiselect("Coverage",sorted(claims.target.unique()),default=[])
c,p=filter_data(claims,policy,years,entities,counties,coverages)
premium=p.Premium.sum(); losses=p.BCClaim.sum(); freq=p.Freq.sum(); severity=losses/freq if freq else 0; ratio=losses/premium if premium else 0
cards=st.columns(6); vals=[("Total Premium",f"${premium/1e6:,.1f}M","#2563EB"),("Coverage Exposure",f"${p.BCcov.sum()/1e9:,.1f}B","#06B6D4"),("Historical Claims",f"${losses/1e6:,.1f}M","#DC2626"),("Loss Ratio",f"{ratio:.1%}","#F59E0B"),("Claim Frequency",f"{freq:,.0f}","#7C3AED"),("Avg Severity",f"${severity:,.0f}","#0F766E")]
for col,(l,v,a) in zip(cards,vals):
    with col: metric_card(l,v,a)
y=p.groupby('Year',as_index=False).agg(Premium=('Premium','sum'),BCClaim=('BCClaim','sum'),BCcov=('BCcov','sum'),Freq=('Freq','sum')); y['LossRatio']=y.BCClaim/y.Premium
left,right=st.columns([1.35,1])
with left: st.plotly_chart(style_fig(px.area(y,x='Year',y=['Premium','BCClaim'],title='Premium and historical claim trajectory',color_discrete_sequence=['#2563EB','#DC2626'])),use_container_width=True)
with right: st.plotly_chart(style_fig(px.line(y,x='Year',y='LossRatio',markers=True,title='Loss ratio by year',color_discrete_sequence=['#F59E0B'])),use_container_width=True)
left,right=st.columns(2)
with left:
    cov=c.groupby(['target','CoverageLabel'],as_index=False).agg(ClaimAmount=('Claim','sum'),Records=('claim_row_id','count')).sort_values('ClaimAmount')
    st.plotly_chart(style_fig(px.bar(cov,x='ClaimAmount',y='CoverageLabel',orientation='h',color='target',title='Claim amount by coverage category')),use_container_width=True)
with right:
    county=c.groupby('county',as_index=False).agg(ClaimAmount=('Claim','sum')).nlargest(12,'ClaimAmount').sort_values('ClaimAmount')
    st.plotly_chart(style_fig(px.bar(county,x='ClaimAmount',y='county',orientation='h',title='Top counties by claim amount',color='ClaimAmount',color_continuous_scale=['#BEE3F8','#2563EB'])),use_container_width=True)
