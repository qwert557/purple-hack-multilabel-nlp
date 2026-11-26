# ИИ-система выявления дополнительных категорий услуг и разделения объявлений (IT Purple Hack, кейс Avito)

Командный проект, разработанный в рамках хакатона **«IT Purple Hack» (трек Avito)**. Система решает задачу автоматического анализа текстов объявлений в сфере ремонта и строительства: определяет скрытые дополнительные категории услуг (`targetSplitMcIds`), принимает решение о необходимости разбиения объявления (`shouldSplit`) и генерирует черновики отдельных объявлений с помощью LLM.

---

## Мой вклад в проект
Я отвечала за **R&D и разработку ML-модели многометочной классификации (Multi-label Classification)**, включая проектирование архитектуры, лингвистический Feature Engineering и сравнение классических ансамблей с нейросетевыми трансформерами:

1. **Проектирование гибридного классификатора (`EnsembleModel` в `avito/src/ml/ensemble_model.py`):**
   * Разработала мультимодальное признаковое пространство, объединяющее плотные семантические эмбеддинги (`Sentence-Transformers`), разреженные n-граммные признаки (`TfidfVectorizer`, `ngram_range=(1,2)`), бинарные индикаторы базовых категорий (`targetDetectedMcIds`) и доменные лингвистические признаки.
   * Создала набор регулярных выражений (`avito/src/utils/markers.py`) для детекции речевых маркеров раздельного выполнения работ (*«по отдельности»*, *«каждый этап»*), комплексных услуг (*«под ключ»*) и узкой специализации.
   * Реализовала обучение ансамбля бинарных классификаторов `CatBoostClassifier` (стратегия *One-vs-Rest*) с индивидуальным подбором порогов вероятности на валидации для максимизации `F1-score`.

2. **Экспериментальное сравнение с End-to-End трансформерами (Jupyter Notebook):**
   * Провела серию экспериментов по сравнению предобученных языковых моделей (`paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-base`, `RuBERT-base`, `ruBert-base-v2`, `XLM-R-base`).
   * Реализовала на **PyTorch** кастомную двухголовую нейросетевую архитектуру (`FullModel` с `MeanPooling`, `MultiLabelHead` и `ShouldSplitHead`) с функцией потерь **Focal Loss** ($\gamma = 2.0$), `Gradient Checkpointing`, `Mixed Precision (AMP)` и поклассовой калибровкой порогов (`find_per_class_thresholds`).
   * Экспериментально доказала превосходство гибридного подхода (эмбеддинги + TF-IDF + лингвистические маркеры + CatBoost) над сквозным fine-tuning трансформеров на данном объёме размеченных данных.

---

## Сравнение моделей и метрики

| Подход / Модель | Признаки / Архитектура | Test F1-micro | Test F1-macro |
| :--- | :--- | :---: | :---: |
| **MiniLM-L12-v2 + CatBoost (Best)** | `Embeddings + TF-IDF + LingFeatures + Binary` | **`0.7033`** | **`0.7020`** |
| **multilingual-e5-base + CatBoost** | `Embeddings + TF-IDF + LingFeatures + Binary` | `0.6826` | `0.6783` |
| **RuBERT-base (sentence) + CatBoost** | `Embeddings + TF-IDF + LingFeatures + Binary` | `0.6581` | `0.6608` |
| **RuBERT-base (End-to-End, Pipeline A)** | `Dual-Head + Focal Loss + Per-class Thr` | `0.4711` | `0.4559` |
| **ruBert-base-v2 (End-to-End)** | `Dual-Head + Focal Loss + Global Thr` | `0.4388` | `0.4205` |
| **XLM-R-base (End-to-End)** | `Dual-Head + Focal Loss + Global Thr` | `0.3626` | `0.3488` |


---

## Структура репозитория

### Мои модули и R&D (классификация категорий и Feature Engineering)
* `avito/src/ml/ensemble_model.py` — продакшен-реализация ансамбля `EnsembleModel` (Sentence-Transformers + TF-IDF + лингвистические признаки + CatBoost с калибровкой порогов).
* `avito/src/utils/markers.py` — словари регулярных выражений для извлечения доменных лингвистических маркеров (`separate_markers`, `complex_markers`, `specialization_markers`).
* `notebooks/experiments_multilabel_nlp.ipynb` — исследовательский ноутбук с полным циклом экспериментов, реализацией Dual-Head трансформеров на PyTorch (`FocalLoss`) и сравнением метрик.

### Остальные модули системы (командная разработка)
* `avito/src/detection/hybrid_detector.py` — первичный гибридный детектор базовых категорий по ключевым фразам и косинусному сходству TF-IDF.
* `avito/src/pipeline/pipeline.py` — сквозной пайплайн `Pipeline`: детекция базовых категорий, предсказание дополнительных услуг, генерация черновиков объявлений через LLM (`Qwen2.5-7B-Instruct`) и расчёт итоговых метрик.
* `avito/src/preprocessing/clean_text.py` — очистка и лемматизация русских текстов (`pymorphy3`, `NLTK`).
* `avito/src/utils/config.py` / `parsing.py` — конфигурация путей и утилиты парсинга списков категорий.
* `avito/main.py` / `avito/prediction.py` — точки входа для запуска инференса на одиночном тексте и пакетной генерации сабмита.

---

## Стек
`Python` • `PyTorch` • `Hugging Face Transformers` • `Sentence-Transformers` • `CatBoost` • `Scikit-learn` • `pymorphy3` • `Pandas` • `NumPy`

## Запуск
```bash
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token"

# Инференс на примере объявления
python avito/main.py

# Пакетный инференс на тестовой выборке
python avito/prediction.py
```
