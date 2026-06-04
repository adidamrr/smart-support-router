import os
from typing import Any

import joblib
from data import load_banking77
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


def load_data() -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    data = load_banking77()
    train_labels = [data.label_names[label_id] for label_id in data.train_label_ids]
    test_labels = [data.label_names[label_id] for label_id in data.test_label_ids]
    return data.train_texts, train_labels, data.test_texts, test_labels, data.label_names


def vectorize_text(train_texts: list[str], test_texts: list[str]) -> tuple[Any, Any, TfidfVectorizer]:
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    train_features = tfidf.fit_transform(train_texts)
    test_features = tfidf.transform(test_texts)

    return train_features, test_features, tfidf


def train_model(train_features: Any, train_labels: list[str]) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(train_features, train_labels)
    return model


def print_metrics(
    model: LogisticRegression,
    test_features: Any,
    test_labels: list[str],
    label_names: list[str],
) -> None:
    pred = model.predict(test_features)
    accuracy = accuracy_score(test_labels, pred)
    f1 = f1_score(test_labels, pred, average="macro", zero_division=0)

    print(f"intent accuracy: {accuracy:.4f}")
    print(f"intent f1_macro: {f1:.4f}")
    print(
        classification_report(
            test_labels,
            pred,
            labels=label_names,
            zero_division=0,
        )
    )


def save_model(model: LogisticRegression, tfidf: TfidfVectorizer) -> None:
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/intent_model.joblib")
    joblib.dump(tfidf, "models/tfidf.joblib")


def main() -> None:
    train_texts, train_labels, test_texts, test_labels, label_names = load_data()

    print(f"Train size: {len(train_texts)}")
    print(f"Test size: {len(test_texts)}")
    print(f"Num classes: {len(label_names)}")

    train_features, test_features, tfidf = vectorize_text(train_texts, test_texts)
    model = train_model(train_features, train_labels)

    print_metrics(model, test_features, test_labels, label_names)
    save_model(model, tfidf)


if __name__ == "__main__":
    main()
