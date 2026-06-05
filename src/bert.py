import argparse
import json
import random
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from data import load_banking77
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "BANKING77"
EPOCHS = 3
DEFAULT_OUTPUT_DIR = "artifacts/bert"


class BertIntentClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        return self.intent_head(cls_embedding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a BERT model")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_texts(
    tokenizer: Any,
    texts: list[str],
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoding = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"]


def create_dataloader(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: list[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> Any:
    if scheduler_name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    true_labels: list[int] = []
    pred_labels: list[int] = []

    for input_ids, attention_mask, labels in tqdm(train_loader, desc="Training", leave=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=1)
        true_labels.extend(labels.detach().cpu().numpy().tolist())
        pred_labels.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = running_loss / len(train_loader.dataset)
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_names: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    running_loss = 0.0
    true_labels: list[int] = []
    pred_labels: list[int] = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            running_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1)
            true_labels.extend(labels.detach().cpu().numpy().tolist())
            pred_labels.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    report = classification_report(
        true_labels,
        pred_labels,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
    }

    return metrics, report


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def log_mlflow_artifacts(output_dir: Path) -> None:
    mlflow.log_artifact(str(output_dir / "metrics.json"))
    mlflow.log_artifact(str(output_dir / "classification_report.json"))
    mlflow.log_artifact(str(output_dir / "model.pt"))
    mlflow.log_artifact(str(output_dir / "label_names.json"))
    mlflow.log_artifacts(str(output_dir / "tokenizer"), artifact_path="tokenizer")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("smart-support-router")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_banking77()
    num_classes = len(data.label_names)

    print(f"Train size: {len(data.train_texts)}")
    print(f"Test size: {len(data.test_texts)}")
    print(f"Num classes: {num_classes}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_input_ids, train_attention_mask = tokenize_texts(tokenizer, data.train_texts, args.max_len)
    test_input_ids, test_attention_mask = tokenize_texts(tokenizer, data.test_texts, args.max_len)

    train_loader = create_dataloader(
        train_input_ids,
        train_attention_mask,
        data.train_label_ids,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = create_dataloader(
        test_input_ids,
        test_attention_mask,
        data.test_label_ids,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = BertIntentClassifier(
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = create_scheduler(
        scheduler_name=args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_test_f1 = -1.0
    best_metrics: dict[str, Any] = {}
    best_report: dict[str, Any] = {}

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "dataset_name": DATASET_NAME,
                "epochs": EPOCHS,
                "learning_rate": args.lr,
                "dropout": args.dropout,
                "scheduler": args.scheduler,
            }
        )

        for epoch in range(1, EPOCHS + 1):
            train_metrics = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            test_metrics, test_report = evaluate(
                model=model,
                data_loader=test_loader,
                criterion=criterion,
                device=device,
                label_names=data.label_names,
            )

            print(f"\nEpoch {epoch}/{EPOCHS}")
            print(f"Train loss: {train_metrics['loss']:.4f}")
            print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
            print(f"Train f1_macro: {train_metrics['f1_macro']:.4f}")
            print(f"Test loss: {test_metrics['loss']:.4f}")
            print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
            print(f"Test f1_macro: {test_metrics['f1_macro']:.4f}")

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1_macro": train_metrics["f1_macro"],
                    "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_f1_macro": test_metrics["f1_macro"],
                },
                step=epoch,
            )

            if test_metrics["f1_macro"] > best_test_f1:
                best_test_f1 = test_metrics["f1_macro"]
                torch.save(model.state_dict(), args.output_dir / "model.pt")
                best_report = test_report
                best_metrics = {
                    "best_test_accuracy": test_metrics["accuracy"],
                    "best_test_f1_macro": test_metrics["f1_macro"],
                    "best_test_loss": test_metrics["loss"],
                    "best_train_accuracy": train_metrics["accuracy"],
                    "best_train_f1_macro": train_metrics["f1_macro"],
                    "best_train_loss": train_metrics["loss"],
                    "best_epoch": epoch,
                    "model_name": MODEL_NAME,
                    "dataset_name": DATASET_NAME,
                    "max_len": args.max_len,
                    "batch_size": args.batch_size,
                    "epochs": EPOCHS,
                    "lr": args.lr,
                    "dropout": args.dropout,
                    "scheduler": args.scheduler,
                    "seed": args.seed,
                }

        tokenizer.save_pretrained(args.output_dir / "tokenizer")
        save_json(data.label_names, args.output_dir / "label_names.json")
        save_json(best_metrics, args.output_dir / "metrics.json")
        save_json(best_report, args.output_dir / "classification_report.json")

        mlflow.log_metrics(
            {
                "accuracy": best_metrics["best_test_accuracy"],
                "f1_macro": best_metrics["best_test_f1_macro"],
                "best_epoch": best_metrics["best_epoch"],
                "test_loss": best_metrics["best_test_loss"],
            }
        )
        log_mlflow_artifacts(args.output_dir)

    print(f"\nSaved best model and artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
