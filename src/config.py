from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "CLAIMLEVEL.csv"
POLICY_PATH = ROOT / "WiscPropFund.csv"
MIN_CLASS_SIZE = 30
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_K = 5
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"
COLORS = {"navy":"#10243E","blue":"#2563EB","teal":"#0F766E","cyan":"#06B6D4","amber":"#F59E0B","red":"#DC2626","slate":"#64748B"}
RISK_WEIGHTS = {"amount":0.40,"uncertainty":0.20,"mismatch":0.20,"conflict":0.20}
CLAIM_REQUIRED = ["PolicyNum","ClaimNum","Year","ClaimStatus","Claim","Deduct","EntityType","Description","CoverageGroup","CoverageCode","Fire5","CountyCode","county"]
POLICY_REQUIRED = ["PolicyNum","Year","Premium","Deduct","BCcov","Freq","Fire5","NoClaimCredit","EntityType","AlarmCredit","BCClaim"]
COVERAGE_LABELS = {"VF":"Fire, Lightning & Electrical","VS":"Water, Sewer & Service-related","VE":"Vandalism, Theft, Wind & External Damage","Other":"Other / Rare Coverage"}
