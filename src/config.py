"""Central configuration for the Traffic Accident Severity project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Addis_Ababa_city_RTA.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET = "Accident_severity"
TIME_COL = "Time"
NUMERIC_COLS = ["Number_of_vehicles_involved", "Number_of_casualties"]

# Order used for reports/plots (least -> most severe).
CLASS_ORDER = ["Slight Injury", "Serious Injury", "Fatal injury"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
