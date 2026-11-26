from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

RNC_DATASET_RAW = RAW_DIR / "rnc_dataset_markup.csv"
RNC_MIC_KEY_PHRASES = RAW_DIR / "rnc_mic_key_phrases.csv"

RNC_DATASET_CLEANED = PROCESSED_DIR / "rnc_dataset_markup_cleaned.csv"
RNC_DATASET_DETECTED = PROCESSED_DIR / "rnc_dataset_detected_1col.csv"

ENSEMBLE_MODEL_PATH = MODELS_DIR / "ensemble.pkl"

