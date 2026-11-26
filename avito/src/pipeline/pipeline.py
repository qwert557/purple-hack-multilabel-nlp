import pickle
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from avito.src.utils.config import ENSEMBLE_MODEL_PATH
from avito.src.utils.parsing import parse_list_int
from avito.src.detection.hybrid_detector import (
    load_detection_artifacts,
    find_categories_hybrid
)
from avito.src.ml.ensemble_model import EnsembleModel
import os
from openai import OpenAI


class Pipeline:
    def __init__(self):
        self.categories = None
        self.tfidf = None
        self.matrix = None
        self.category_ids = None
        self.model = None
        self.mc_id_to_title = {
            101: "Ремонт квартир и домов под ключ",
            102: "Сантехника",
            103: "Электрика",
            104: "Натяжные потолки",
            105: "Укладка плитки",
            106: "Поклейка обоев",
            107: "Малярные работы",
            108: "Штукатурные работы",
            109: "Напольные покрытия",
            110: "Гипсокартон",
            111: "Демонтажные работы"
        }
        self.use_llm=False
        self.llm = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )

    def load_detection(self):
        self.categories, self.tfidf, self.matrix, self.category_ids = load_detection_artifacts()

    def save_model(self):
        ENSEMBLE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ENSEMBLE_MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self):
        if ENSEMBLE_MODEL_PATH.exists():
            with open(ENSEMBLE_MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            return True
        return False

    def train_model(self, df_markup_path: str):
        df = pd.read_csv(df_markup_path, sep=';')
        self.model = EnsembleModel().fit(df)
        self.save_model()

    def fit(self, df_markup_path: str):
        print("Загружаем detection-артефакты...")
        self.load_detection()

        print("Пробуем загрузить модель...")
        if self.load_model():
            print("Модель загружена.")
        else:
            print("Модель не найдена. Обучаем...")
            self.train_model(df_markup_path)
            print("Модель обучена и сохранена.")

        return self

    def detect_mc_ids(self, text: str):
        return find_categories_hybrid(
            description=text,
            categories=self.categories,
            tfidf=self.tfidf,
            matrix=self.matrix,
            category_ids=self.category_ids
        )

    def generate_draft_text(self, original_text: str, mc_title: str) -> str:
        if not self.use_llm:
            return f"{mc_title}: выполняем работы по направлению."

        prompt = (
            f"Сгенерируй короткий, естественный текст черновика объявления "
            f"для услуги '{mc_title}'. "
            f"Используй информацию из описания: '{original_text}'. "
            f"Текст должен быть 1-2 предложения, без лишней воды. "
        )

        completion = self.llm.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct:together",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4
        )

        return completion.choices[0].message.content.strip()

    def predict(self, text: str):
        detected_mc_ids = self.detect_mc_ids(text)

        result = self.model.predict(text, detected_mc_ids)
        draft_mc_ids = result["draft_mcIds"]

        drafts = []
        for mc_id in draft_mc_ids:
            mc_title = self.mc_id_to_title.get(mc_id, f"Категория {mc_id}")
            draft_text = self.generate_draft_text(text, mc_title)

            drafts.append({
                "mcId": int(mc_id),
                "mcTitle": mc_title,
                "text": draft_text
            })

        return {
            "detectedMcIds": detected_mc_ids,
            "shouldSplit": len(drafts) > 0,
            "drafts": drafts
        }

    def evaluate_f1(self, df_eval: pd.DataFrame):
        y_true_mc = df_eval["targetSplitMcIds"].apply(parse_list_int).tolist()
        y_true_split = df_eval["shouldSplit"].astype(int).tolist()

        y_pred_mc = []
        y_pred_split = []

        for _, row in df_eval.iterrows():
            pred = self.predict(row["description"])
            y_pred_mc.append([int(d["mcId"]) for d in pred["drafts"]])
            y_pred_split.append(int(pred["shouldSplit"]))

        mlb = MultiLabelBinarizer()
        Y_true_mc = mlb.fit_transform(y_true_mc)
        Y_pred_mc = mlb.transform(y_pred_mc)

        f1_micro = f1_score(Y_true_mc, Y_pred_mc, average="micro", zero_division=0)
        shouldSplit_acc = accuracy_score(y_true_split, y_pred_split)

        return {
            "f1_micro": f1_micro,
            "shouldSplit_accuracy": shouldSplit_acc
        }
