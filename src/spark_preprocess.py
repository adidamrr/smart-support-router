import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MISSING_JAVA_MESSAGE = "Java runtime not found. Install Java before running Spark preprocessing."

RAW_DATA_FILES = {
    "train": {
        "url": (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            "master/banking_data/train.csv"
        ),
        "path": RAW_DATA_DIR / "train.csv",
    },
    "test": {
        "url": (
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
            "master/banking_data/test.csv"
        ),
        "path": RAW_DATA_DIR / "test.csv",
    },
}


def ensure_java_available() -> None:
    try:
        subprocess.run(["java", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(MISSING_JAVA_MESSAGE) from exc


def download_raw_data() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split, file_info in RAW_DATA_FILES.items():
        path = file_info["path"]
        if path.exists():
            print(f"Raw {split} data already exists: {path}")
            continue

        print(f"Downloading raw {split} data to: {path}")
        urlretrieve(file_info["url"], path)


def read_raw_split(spark: SparkSession, split: str) -> DataFrame:
    return spark.read.csv(
        str(RAW_DATA_FILES[split]["path"]),
        header=True,
        inferSchema=False,
        multiLine=True,
        escape='"',
    )


def clean_split(df: DataFrame) -> DataFrame:
    cleaned = df.select(
        F.trim(F.col("text").cast("string")).alias("text"),
        F.trim(F.col("category").cast("string")).alias("intent"),
    )
    return cleaned.filter(
        F.col("text").isNotNull()
        & F.col("intent").isNotNull()
        & (F.col("text") != "")
        & (F.col("intent") != "")
    )


def count_text_duplicates(df: DataFrame) -> int:
    return df.count() - df.select("text").distinct().count()


def intent_distribution(df: DataFrame, split: str) -> pd.DataFrame:
    distribution = (
        df.groupBy("intent")
        .count()
        .withColumn("split", F.lit(split))
        .select("split", "intent", "count")
        .orderBy("split", "intent")
    )
    return distribution.toPandas()


def save_processed_split(df: DataFrame, split: str) -> None:
    output_path = PROCESSED_DATA_DIR / f"{split}.parquet"
    df.write.mode("overwrite").parquet(str(output_path))


def main() -> None:
    ensure_java_available()
    download_raw_data()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("smart-support-router-preprocessing")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        train_df = clean_split(read_raw_split(spark, "train"))
        test_df = clean_split(read_raw_split(spark, "test"))

        print(f"Train rows: {train_df.count()}")
        print(f"Test rows: {test_df.count()}")
        print(f"Train duplicate texts: {count_text_duplicates(train_df)}")
        print(f"Test duplicate texts: {count_text_duplicates(test_df)}")

        save_processed_split(train_df, "train")
        save_processed_split(test_df, "test")

        distribution_df = pd.concat(
            [
                intent_distribution(train_df, "train"),
                intent_distribution(test_df, "test"),
            ],
            ignore_index=True,
        )
        distribution_df.to_csv(PROCESSED_DATA_DIR / "intent_distribution.csv", index=False)

        print(f"Saved processed data to: {PROCESSED_DATA_DIR}")
    finally:
        spark.stop()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)
