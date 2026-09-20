import streamlit as st
import numpy as np, plotly.express as px
from src.ui import page,metric_card,style_fig
from src.state import datasets
page("Historical Portfolio Analytics","Interactive policy, exposure, deductible, frequency and loss analysis.")
_,_,claims,p=datasets(); years=st.sidebar.slider("Year range",int(p.Year.min()),int(p.Year.max()),(int(p.Year.min()),int(p.Year.max())),key='histyears'); p=p[p.Year.between(*years)].copy()
p["LossRatio"]=np.where(p.Premium>0,p.BCClaim/p.Premium,0); p["ClaimStatus"]=np.where(p.BCClaim>0,"Claim activity","No claim activity")
cards=st.columns(4)
for col,(l,v,a) in zip(cards,[("Active policy-years",f"{len(p):,}","#2563EB"),("Zero-claim share",f"{p.BCClaim.eq(0).mean():.1%}","#0F766E"),("Median deductible",f"${p.Deduct.median():,.0f}","#7C3AED"),("Largest historical loss",f"${p.BCClaim.max()/1e6:,.1f}M","#DC2626")]):
    with col: metric_card(l,v,a)
tabs=st.tabs(["Portfolio Trends","Risk & Distribution","Policy Drill-down"])
with tabs[0]:
    yr=p.groupby('Year',as_index=False).agg(Premium=('Premium','sum'),BCClaim=('BCClaim','sum'),BCcov=('BCcov','sum'),Freq=('Freq','sum'))
    a,b=st.columns(2); a.plotly_chart(style_fig(px.line(yr,x='Year',y='BCcov',markers=True,title='Coverage exposure trend',color_discrete_sequence=['#06B6D4'])),use_container_width=True); b.plotly_chart(style_fig(px.bar(yr,x='Year',y='Freq',title='Claim frequency by year',color='Freq',color_continuous_scale=['#C4B5FD','#6D28D9'])),use_container_width=True)
with tabs[1]:
    a,b=st.columns(2); a.plotly_chart(style_fig(px.scatter(p,x='Premium',y='BCClaim',size='BCcov',color='ClaimStatus',hover_data=['PolicyNum','Year'],title='Premium vs historical claim amount',color_discrete_map={'Claim activity':'#DC2626','No claim activity':'#94A3B8'})),use_container_width=True); b.plotly_chart(style_fig(px.histogram(p,x='Freq',color='ClaimStatus',title='Claim frequency distribution',barmode='overlay')),use_container_width=True)
    top=p.nlargest(15,'BCClaim').copy(); top['PolicyYear']=top.PolicyNum.astype(str)+' · '+top.Year.astype(str); st.plotly_chart(style_fig(px.bar(top.sort_values('BCClaim'),x='BCClaim',y='PolicyYear',orientation='h',title='Highest-loss policy-years',color='LossRatio',color_continuous_scale='OrRd'),500),use_container_width=True)
with tabs[2]:
    selected=st.selectbox('Select a policy',sorted(p.PolicyNum.unique())); one=p[p.PolicyNum.eq(selected)].sort_values('Year'); st.plotly_chart(style_fig(px.line(one,x='Year',y=['Premium','BCClaim'],markers=True,title=f'Policy {selected}: premium and claims')),use_container_width=True); st.dataframe(one[['Year','Premium','Deduct','BCcov','Freq','BCClaim','NoClaimCredit','AlarmCredit']],hide_index=True,use_container_width=True)
