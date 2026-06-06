import importlib
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient


class FakeIntentPredictor:
    def predict(self, text: str) -> dict[str, Any]:
        return {
            "text": text,
            "intent": "atm_support",
            "confidence": 0.91,
            "suggested_reply": "Please share the ATM location, date, and transaction details.",
            "top_intents": [
                {"intent": "atm_support", "confidence": 0.91},
                {"intent": "cash_withdrawal_not_recognised", "confidence": 0.05},
            ],
        }


@pytest.fixture()
def api_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    fake_database = types.ModuleType("src.database")
    fake_database.init_db = lambda: None
    fake_database.save_prediction = lambda **kwargs: None
    fake_database.get_recent_predictions = lambda limit: []

    fake_inference = types.ModuleType("src.inference")
    fake_inference.IntentPredictor = FakeIntentPredictor

    monkeypatch.setitem(sys.modules, "src.database", fake_database)
    monkeypatch.setitem(sys.modules, "src.inference", fake_inference)
    sys.modules.pop("src.api", None)

    module = importlib.import_module("src.api")
    yield module

    sys.modules.pop("src.api", None)


def test_health_returns_ok(api_module: Any) -> None:
    with TestClient(api_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_payload(api_module: Any) -> None:
    with TestClient(api_module.app) as client:
        response = client.post(
            "/predict",
            json={"text": "I cannot withdraw cash from the ATM"},
        )

    payload = response.json()

    assert response.status_code == 200
    assert payload["text"] == "I cannot withdraw cash from the ATM"
    assert payload["intent"] == "atm_support"
    assert payload["confidence"] == 0.91
    assert isinstance(payload["suggested_reply"], str)
    assert payload["top_intents"][0] == {"intent": "atm_support", "confidence": 0.91}
