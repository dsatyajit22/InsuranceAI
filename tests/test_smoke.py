from src.data import load_data
from src.model import IntelligenceEngine

def test_smoke():
    _,_,claims,policy=load_data(); assert len(claims)>0 and len(policy)>0
    result=IntelligenceEngine(claims).assess('water damage from burst pipe',9000,'VS',5)
    assert 0<=result['risk_score']<=100 and len(result['matches'])==5
