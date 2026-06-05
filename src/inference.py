import json
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from .templates import get_suggested_reply
from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel


DEFAULT_ARTIFACTS_DIR = Path("artifacts/bert")
DEFAULT_MAX_LEN = 128
DEFAULT_DROPOUT = 0.2


class BertIntentClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.bert = DistilBertModel(DistilBertConfig())
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        return self.intent_head(cls_embedding)


class IntentPredictor:
    def __init__(
        self,
        artifacts_dir: Union[str, Path] = DEFAULT_ARTIFACTS_DIR,
        max_len: int = DEFAULT_MAX_LEN,
        dropout: float = DEFAULT_DROPOUT,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.max_len = max_len
        self.dropout = dropout
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.label_names = self._load_label_names()
        self.tokenizer = AutoTokenizer.from_pretrained(self.artifacts_dir / "tokenizer")
        self.model = BertIntentClassifier(
            num_classes=len(self.label_names),
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(self.artifacts_dir / "model.pt", map_location=self.device)
        )
        self.model.eval()

    def _load_label_names(self) -> list[str]:
        label_names_path = self.artifacts_dir / "label_names.json"
        with label_names_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def predict(self, text: str, top_k: int = 3) -> dict[str, Any]:
        top_k = max(1, min(top_k, len(self.label_names)))
        encoding = self.tokenizer(
            [str(text)],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = torch.topk(probabilities, k=top_k, dim=1)

        confidence_values = confidences.squeeze(0).detach().cpu().tolist()
        prediction_ids = predictions.squeeze(0).detach().cpu().tolist()
        top_intents = [
            {
                "intent": self.label_names[label_id],
                "confidence": float(confidence),
            }
            for label_id, confidence in zip(prediction_ids, confidence_values)
        ]

        best_intent = top_intents[0]["intent"]
        best_confidence = top_intents[0]["confidence"]

        return {
            "text": text,
            "intent": best_intent,
            "confidence": best_confidence,
            "suggested_reply": get_suggested_reply(best_intent),
            "top_intents": top_intents,
        }
