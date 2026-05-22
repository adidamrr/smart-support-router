from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "tickets_ru_10000.csv"
MODELS_DIR = PROJECT_DIR / "models"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df["ticket_text_ru"]
    y_queue = df["queue"]
    y_priority = df["priority"]

    encoder_queue = LabelEncoder()
    encoder_queue.fit(y_queue)
    y_queue = encoder_queue.transform(y_queue)

    encoder_priority = LabelEncoder()
    encoder_priority.fit(y_priority)
    y_priority = encoder_priority.transform(y_priority)

    X_train, X_test, y_queue_train, y_queue_test, y_priority_train, y_priority_test = train_test_split(
        X,
        y_queue,
        y_priority,
        test_size=0.2,
        random_state=42,
    )

    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    tfidf.fit(X_train)
    train_text = tfidf.transform(X_train)
    test_text = tfidf.transform(X_test)

    queue_model = LogisticRegression(class_weight="balanced")
    queue_model.fit(train_text, y_queue_train)

    pred_queue = queue_model.predict(test_text)
    queue_accuracy = accuracy_score(y_queue_test, pred_queue)
    queue_f1 = f1_score(y_queue_test, pred_queue, average="macro")

    priority_model = LogisticRegression(class_weight="balanced")
    priority_model.fit(train_text, y_priority_train)

    pred_priority = priority_model.predict(test_text)
    priority_accuracy = accuracy_score(y_priority_test, pred_priority)
    priority_f1 = f1_score(y_priority_test, pred_priority, average="macro")

    print(f"queue accuracy: {queue_accuracy:.4f}")
    print(f"queue f1 macro: {queue_f1:.4f}")
    print(f"priority accuracy: {priority_accuracy:.4f}")
    print(f"priority f1 macro: {priority_f1:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(queue_model, MODELS_DIR / "queue_model.joblib")
    joblib.dump(priority_model, MODELS_DIR / "priority_model.joblib")
    joblib.dump(tfidf, MODELS_DIR / "tfidf.joblib")
    joblib.dump(encoder_queue, MODELS_DIR / "encoder_queue.joblib")
    joblib.dump(encoder_priority, MODELS_DIR / "encoder_priority.joblib")


if __name__ == "__main__":
    main()
