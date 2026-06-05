# Airflow training pipeline

This Airflow setup orchestrates the Smart Support Router training pipeline:

1. Spark preprocessing
2. BERT training
3. Artifact check

The DAG does not contain ML code. It runs the existing project scripts from the main `.venv`.

## Requirements

Install Java before running the Spark preprocessing task:

```bash
brew install openjdk@17
export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
java -version
```

Install project dependencies in the main environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Install Airflow

Use a separate Airflow environment so Airflow dependencies do not change the project `.venv`.

```bash
python3 -m venv .venv-airflow
source .venv-airflow/bin/activate
pip install "apache-airflow==2.10.5" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.9.txt"
```

## Run Airflow locally

Run these commands from the project root:

```bash
source .venv-airflow/bin/activate
export AIRFLOW_HOME="$(pwd)/airflow"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
airflow db init
airflow dags list
```

Trigger the training pipeline manually:

```bash
airflow dags trigger smart_support_training_pipeline
```

The DAG is manual only because BERT training is expensive and should not run on a schedule by default.

## Outputs

After a successful run, the project should contain:

```text
data/processed/train.parquet
data/processed/test.parquet
data/processed/intent_distribution.csv
artifacts/bert/model.pt
artifacts/bert/tokenizer/
artifacts/bert/label_names.json
artifacts/bert/metrics.json
artifacts/bert/classification_report.json
```
