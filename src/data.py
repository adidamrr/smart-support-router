from dataclasses import dataclass

from datasets import load_dataset


DATA_FILES = {
    "train": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
        "master/banking_data/train.csv"
    ),
    "test": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
        "master/banking_data/test.csv"
    ),
}


@dataclass(frozen=True)
class Banking77Data:
    train_texts: list[str]
    train_label_ids: list[int]
    test_texts: list[str]
    test_label_ids: list[int]
    label_names: list[str]


def load_banking77() -> Banking77Data:
    dataset = load_dataset("csv", data_files=DATA_FILES)

    train_texts = [str(text).strip() for text in dataset["train"]["text"]]
    test_texts = [str(text).strip() for text in dataset["test"]["text"]]

    train_categories = [str(label).strip() for label in dataset["train"]["category"]]
    test_categories = [str(label).strip() for label in dataset["test"]["category"]]

    label_names = sorted(set(train_categories) | set(test_categories))
    label_to_id = {label: label_id for label_id, label in enumerate(label_names)}

    train_label_ids = [label_to_id[label] for label in train_categories]
    test_label_ids = [label_to_id[label] for label in test_categories]

    return Banking77Data(
        train_texts=train_texts,
        train_label_ids=train_label_ids,
        test_texts=test_texts,
        test_label_ids=test_label_ids,
        label_names=label_names,
    )
