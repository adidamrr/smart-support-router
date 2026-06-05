from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PARQUET_PATH = PROCESSED_DATA_DIR / "train.parquet"
TEST_PARQUET_PATH = PROCESSED_DATA_DIR / "test.parquet"
MISSING_PROCESSED_DATA_MESSAGE = "Processed data not found. Run python src/spark_preprocess.py before training."


@dataclass(frozen=True)
class Banking77Data:
    train_texts: list[str]
    train_label_ids: list[int]
    test_texts: list[str]
    test_label_ids: list[int]
    label_names: list[str]


def load_banking77() -> Banking77Data:
    if not TRAIN_PARQUET_PATH.exists() or not TEST_PARQUET_PATH.exists():
        raise FileNotFoundError(MISSING_PROCESSED_DATA_MESSAGE)

    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    test_df = pd.read_parquet(TEST_PARQUET_PATH)

    train_texts = train_df["text"].astype(str).str.strip().tolist()
    test_texts = test_df["text"].astype(str).str.strip().tolist()

    train_intents = train_df["intent"].astype(str).str.strip().tolist()
    test_intents = test_df["intent"].astype(str).str.strip().tolist()

    label_names = sorted(set(train_intents) | set(test_intents))
    label_to_id = {label: label_id for label_id, label in enumerate(label_names)}

    train_label_ids = [label_to_id[label] for label in train_intents]
    test_label_ids = [label_to_id[label] for label in test_intents]

    return Banking77Data(
        train_texts=train_texts,
        train_label_ids=train_label_ids,
        test_texts=test_texts,
        test_label_ids=test_label_ids,
        label_names=label_names,
    )
