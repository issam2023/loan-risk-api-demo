import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MODEL_PATH", "models/loan_default_model_v1.joblib")
os.environ.setdefault("METADATA_PATH", "models/loan_default_model_v1.json")
os.environ.setdefault("MAX_BATCH", "500")
