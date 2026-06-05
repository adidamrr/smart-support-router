from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"


def check_artifacts() -> None:
    required_paths = [
        PROJECT_ROOT / "artifacts" / "bert" / "model.pt",
        PROJECT_ROOT / "artifacts" / "bert" / "tokenizer",
        PROJECT_ROOT / "artifacts" / "bert" / "label_names.json",
        PROJECT_ROOT / "artifacts" / "bert" / "metrics.json",
        PROJECT_ROOT / "artifacts" / "bert" / "classification_report.json",
    ]
    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing training artifacts:\n{missing}")


with DAG(
    dag_id="smart_support_training_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["smart-support-router", "training"],
) as dag:
    spark_preprocess = BashOperator(
        task_id="spark_preprocess",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/spark_preprocess.py",
    )

    train_bert = BashOperator(
        task_id="train_bert",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} src/bert.py",
    )

    check_training_artifacts = PythonOperator(
        task_id="check_artifacts",
        python_callable=check_artifacts,
    )

    spark_preprocess >> train_bert >> check_training_artifacts
