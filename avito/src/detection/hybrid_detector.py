import pickle
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from avito.src.utils.config import MODELS_DIR, RNC_MIC_KEY_PHRASES

TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.npz"
CATEGORIES_PATH = MODELS_DIR / "categories.pkl"
CATEGORY_IDS_PATH = MODELS_DIR / "category_ids.pkl"


def prepare_categories(df_keys):
    categories = {}
    category_texts = []
    category_ids = []

    for _, row in df_keys.iterrows():
        mc_id = int(row['mcId'])
        mc_title = row['mcTitle']
        description = row.get('description', '')
        phrases = [p.strip().lower() for p in str(row['keyPhrases']).split(';') if p.strip()]

        categories[mc_id] = {
            'title': mc_title,
            'key_phrases': phrases
        }

        category_text = f"{mc_title} {' '.join(phrases)} {description}"
        category_texts.append(category_text)
        category_ids.append(mc_id)

    return categories, category_texts, category_ids


def build_tfidf_cached(category_texts):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if TFIDF_VECTORIZER_PATH.exists() and TFIDF_MATRIX_PATH.exists():
        with open(TFIDF_VECTORIZER_PATH, "rb") as f:
            tfidf = pickle.load(f)
        matrix = sparse.load_npz(TFIDF_MATRIX_PATH)
        return tfidf, matrix

    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    matrix = tfidf.fit_transform(category_texts)

    with open(TFIDF_VECTORIZER_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    sparse.save_npz(TFIDF_MATRIX_PATH, matrix)

    return tfidf, matrix


def find_by_key_phrases(description, categories):
    if not isinstance(description, str) or len(description.strip()) < 10:
        return []

    desc_lower = description.lower()
    found = []

    for mc_id, cat_info in categories.items():
        for phrase in cat_info['key_phrases']:
            if phrase in desc_lower:
                found.append(mc_id)
                break

    return found


def find_by_tfidf(description, tfidf, matrix, category_ids, top_k=5, threshold=0.3):
    if not isinstance(description, str) or len(description.strip()) < 5:
        return []

    desc_vec = tfidf.transform([description])
    sims = cosine_similarity(desc_vec, matrix)[0]

    top_idx = sims.argsort()[-top_k:][::-1]

    result = []
    for i in top_idx:
        if sims[i] >= threshold:
            result.append(category_ids[i])

    return result


def find_categories_hybrid(description, categories, tfidf, matrix, category_ids,
                           key_threshold=0.3, tfidf_top_k=5):

    if not isinstance(description, str) or len(description.strip()) < 10:
        return []

    key_ids = find_by_key_phrases(description, categories)

    tfidf_ids = find_by_tfidf(description, tfidf, matrix, category_ids,
                              top_k=tfidf_top_k, threshold=key_threshold)

    return sorted(set(key_ids + tfidf_ids))


def load_detection_artifacts():
    df_keys = pd.read_csv(RNC_MIC_KEY_PHRASES)

    if CATEGORIES_PATH.exists():
        with open(CATEGORIES_PATH, "rb") as f:
            categories = pickle.load(f)
        with open(CATEGORY_IDS_PATH, "rb") as f:
            category_ids = pickle.load(f)
    else:
        categories, cat_texts, category_ids = prepare_categories(df_keys)
        with open(CATEGORIES_PATH, "wb") as f:
            pickle.dump(categories, f)
        with open(CATEGORY_IDS_PATH, "wb") as f:
            pickle.dump(category_ids, f)

    _, cat_texts, _ = prepare_categories(df_keys)
    tfidf, matrix = build_tfidf_cached(cat_texts)

    return categories, tfidf, matrix, category_ids
