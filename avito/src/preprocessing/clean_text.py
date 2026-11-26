import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
from avito.src.utils.config import RNC_DATASET_RAW, RNC_DATASET_CLEANED, PROCESSED_DIR

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
stop_words.discard('не')
stop_words.discard('ни')
morph = MorphAnalyzer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^а-яА-ЯёЁ\s]', ' ', text.lower())
    words = text.split()
    cleaned_words = [
        morph.parse(word)[0].normal_form
        for word in words
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(cleaned_words)

def run_cleaning(input_path=None, output_path=None):
    input_path = input_path or RNC_DATASET_RAW
    output_path = output_path or RNC_DATASET_CLEANED
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep=';')
    df['cleaned_description'] = df['description'].apply(clean_text)
    df.to_csv(output_path, index=False)
    return df
