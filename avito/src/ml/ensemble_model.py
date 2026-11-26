import numpy as np
import pandas as pd
import re
import logging
from avito.src.utils.markers import separate_markers, complex_markers, specialization_markers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

from avito.src.utils.parsing import parse_list_int
from avito.src.utils.config import MODELS_DIR

ST_MODEL_DIR = MODELS_DIR / "st_model"
ST_EMB_PATH = MODELS_DIR / "st_embeddings.npy"


class EnsembleModel:
    def __init__(self):
        self.all_mc_ids = None
        self.classes_used = None
        self.mlb = None
        self.tfidf = None
        self.st_model = None
        self.models_ens = []
        self.thresholds_ens = []

    def extract_linguistic_features(self, text: str):
        text_low = text.lower()

        f_separate = int(any(re.search(p, text_low) for p in separate_markers))
        #f_complex = int(any(re.search(p, text_low) for p in complex_markers))
        f_specialization = int(any(re.search(p, text_low) for p in specialization_markers))

        f_many_commas = int(text_low.count(",") >= 3)
        f_and = int(" и " in text_low)

        return [f_separate, f_specialization, f_many_commas, f_and]

    def _make_binary_features(self, detected):
        detected = set(detected)
        return [1 if mc in detected else 0 for mc in self.all_mc_ids]

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df["detected"] = df["targetDetectedMcIds"].apply(parse_list_int)
        df["target"]   = df["targetSplitMcIds"].apply(parse_list_int)

        all_mc_ids = sorted(
            set(mc for lst in df["detected"] for mc in lst) |
            set(mc for lst in df["target"]   for mc in lst)
        )
        if 101 in all_mc_ids:
            all_mc_ids.remove(101)
        self.all_mc_ids = all_mc_ids

        ling_features = np.vstack(df["description"].apply(self.extract_linguistic_features))

        mlb = MultiLabelBinarizer()
        Y_full = mlb.fit_transform(df["target"])
        classes_full = list(mlb.classes_)

        if 101 in classes_full:
            idx_101 = classes_full.index(101)
            mask_keep = [i for i in range(len(classes_full)) if i != idx_101]
            Y = Y_full[:, mask_keep]
            classes_used = [c for c in classes_full if c != 101]
        else:
            Y = Y_full
            classes_used = classes_full

        self.mlb = mlb
        self.classes_used = classes_used

        X_bin = np.vstack(df["detected"].apply(self._make_binary_features))

        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3)
        X_text_tfidf = self.tfidf.fit_transform(df["description"]).toarray()

        if ST_MODEL_DIR.exists():
            self.st_model = SentenceTransformer(str(ST_MODEL_DIR))
        else:
            self.st_model = SentenceTransformer("cointegrated/rubert-tiny2")
            self.st_model.save(str(ST_MODEL_DIR))

        if ST_EMB_PATH.exists():
            X_text_st = np.load(ST_EMB_PATH)
        else:
            logging.basicConfig(level=logging.INFO)
            logging.getLogger("sentence_transformers").setLevel(logging.INFO)
            X_text_st = self.st_model.encode(df["description"].tolist(), show_progress_bar=True)
            np.save(ST_EMB_PATH, X_text_st)

        X_ling = ling_features
        X_ens = np.hstack([X_bin, X_ling, X_text_tfidf, X_text_st])

        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X_ens, Y, test_size=0.30, random_state=42
        )
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=0.50, random_state=42
        )

        n_classes = Y_train.shape[1]
        self.models_ens = []
        self.thresholds_ens = []

        for cls in range(n_classes):
            y_tr = Y_train[:, cls]
            y_v  = Y_val[:, cls]
            y_te = Y_test[:, cls]

            model = CatBoostClassifier(
                loss_function="Logloss",
                iterations=1500,
                depth=7,
                learning_rate=0.03,
                l2_leaf_reg=4,
            )

            model.fit(
                X_train, y_tr,
                eval_set=[(X_val, y_v), (X_test, y_te)],
                use_best_model=True,
                early_stopping_rounds=100,
                verbose=200
            )

            proba_val = model.predict_proba(X_val)[:, 1]
            best_thr = 0.0
            best_f1 = 0.0

            for thr in np.linspace(0.05, 0.5, 20):
                pred_thr = (proba_val > thr).astype(int)
                f1 = f1_score(y_v, pred_thr, average="binary", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr

            self.models_ens.append(model)
            self.thresholds_ens.append(best_thr)

        return self

    def predict(self, text: str, detected_mc_ids: list[int]):
        x_bin = np.array([self._make_binary_features(detected_mc_ids)])
        x_tfidf = self.tfidf.transform([text]).toarray()
        x_st = self.st_model.encode([text])
        x_ling = np.array([self.extract_linguistic_features(text)])
        x = np.hstack([x_bin, x_ling, x_tfidf, x_st])

        proba = np.zeros(len(self.classes_used))
        for cls, model in enumerate(self.models_ens):
            proba[cls] = model.predict_proba(x)[0, 1]

        predicted_mc_ids = [
            mc for mc, p, thr in zip(self.classes_used, proba, self.thresholds_ens)
            if p > thr
        ]

        return {
            "shouldSplit": len(predicted_mc_ids) > 0,
            "draft_mcIds": predicted_mc_ids
        }
