# InsuranceAI V3

Presentation-ready Streamlit application with four pages:
1. Executive Overview
2. Historical Portfolio Analytics
3. AI Claim Assessment
4. Agent Decision Center

## Run on Windows with an existing virtual environment

```cmd
cd path\to\InsuranceAI_V3
C:\path\to\post_joining\.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Both datasets are de-duplicated in memory before dashboard calculations.
- WiscPropFund is also constrained to one record per PolicyNum + Year.
- CLAIMLEVEL is used for model training and claim analysis.
- WiscPropFund is used for portfolio analytics.
- Historical period: 2006–2010.
- The score is a review-priority indicator, not fraud probability.
- The tool does not approve or reject claims.
