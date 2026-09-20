import streamlit as st
import plotly.graph_objects as go
from .config import COLORS,COVERAGE_LABELS

def page(title,subtitle):
    st.set_page_config(page_title=title,page_icon="🛡️",layout="wide")
    from pathlib import Path
    css=(Path(__file__).resolve().parents[1]/"assets/style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>",unsafe_allow_html=True)
    st.markdown(f"<div class='hero'><div class='eyebrow'>INSURANCE INTELLIGENCE</div><h1>{title}</h1><p>{subtitle}</p></div>",unsafe_allow_html=True)

def metric_card(label,value,accent="#2563EB",note=""):
    st.markdown(f"<div class='metric-card' style='border-top-color:{accent}'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>",unsafe_allow_html=True)

def style_fig(fig,height=410):
    fig.update_layout(height=height,margin=dict(l=30,r=20,t=65,b=35),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#FFFFFF",font=dict(color="#24364B"),title_font=dict(size=18,color="#10243E"),legend=dict(orientation="h",y=1.12,x=0))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor="#E8EEF5")
    return fig

def risk_bar(score):
    fig=go.Figure(go.Indicator(mode="gauge+number",value=score,number={"suffix":" / 100"},gauge={"shape":"bullet","axis":{"range":[0,100]},"bar":{"color":COLORS['navy']},"steps":[{"range":[0,30],"color":"#D6F5EA"},{"range":[30,60],"color":"#FFF0C2"},{"range":[60,100],"color":"#FFD8D8"}]}))
    fig.update_layout(height=145,margin=dict(l=25,r=25,t=25,b=20),paper_bgcolor="rgba(0,0,0,0)"); return fig
