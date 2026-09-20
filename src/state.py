import streamlit as st
from .data import load_data
from .model import IntelligenceEngine
@st.cache_data(show_spinner=False)
def datasets(): return load_data()
@st.cache_resource(show_spinner="Preparing AI intelligence engine...")
def engine():
    _,_,claims,_=datasets(); return IntelligenceEngine(claims)
