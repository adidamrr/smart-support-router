# Smart Support Router

Сервис поддержки банка: принимает текст клиента и определяет intent.

Задача модели:

```text
text -> intent
```

## Запуск

Сначала нужно подготовить данные и обучить модель:

```bash
pip install -r requirements.txt
python src/spark_preprocess.py
python src/bert.py
```

После обучения появится папка:

```text
artifacts/bert/
```

Запуск сервиса:

```bash
docker compose up
```

Открыть:

- интерфейс: http://localhost:8501
- API: http://localhost:8000/docs
- MLflow: http://localhost:5000

Локальный запуск без Docker:

```bash
uvicorn src.api:app --reload
streamlit run ui/streamlit_app.py
mlflow ui
```

## Данные

Используется BANKING77: датасет с банковскими обращениями клиентов.

- вход: `text`
- цель: `intent`
- классов: `77`

Spark-скрипт скачивает данные, чистит их и сохраняет локально:

```bash
python src/spark_preprocess.py
```

## Модели

В проекте есть три подхода:

- TF-IDF baseline;
- custom Transformer в ноутбуке;
- BERT-family fine-tuning.

В production используется BERT, потому что он лучше понимает смысл текста и даёт качество выше baseline.

Текущие метрики BERT:

- accuracy: около `0.899`
- f1_macro: около `0.899`

## Что делает сервис

Пайплайн:

```text
текст клиента
-> BERT
-> intent + confidence
-> шаблонный ответ
-> API
-> сохранение истории в PostgreSQL
-> показ результата в Streamlit
```

Airflow DAG запускает обучение по шагам:

```text
Spark preprocessing -> BERT training -> artifacts check
```

## Модель в Git

Модель не хранится в репозитории.

`model.pt` тяжёлый, поэтому это артефакт, а не код. Перед запуском сервиса модель нужно обучить:

```bash
python src/bert.py
```

После этого сервис будет брать файлы из `artifacts/bert/`.

## Структура

```text
src/          код API, обучения, данных и предсказаний
ui/           Streamlit-интерфейс
airflow/      DAG для обучения
notebooks/    эксперименты
data/         локальные данные
artifacts/    обученная модель
mlruns/       логи MLflow
```
