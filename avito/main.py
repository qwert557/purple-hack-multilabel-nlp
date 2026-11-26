from src.pipeline.pipeline import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline().fit("data/raw/rnc_dataset_markup.csv")

    text = "Делаем ремонт квартир под ключ, а также отдельно выполняем сантехнические и электромонтажные работы."
    print(pipeline.predict(text))
