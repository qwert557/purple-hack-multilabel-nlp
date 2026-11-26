import pandas as pd
import json
from tqdm import tqdm
from src.pipeline.pipeline import Pipeline

df = pd.read_csv("rnc_test.csv")
pipeline = Pipeline().fit("data/raw/rnc_dataset_markup.csv")
predictions = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
    req = json.loads(row["request"])
    desc = req["description"]
    pred = pipeline.predict(desc)
    predictions.append(pred)

df["prediction"] = predictions
df.to_csv("rnc_test_with_predictions.csv", index=False)
