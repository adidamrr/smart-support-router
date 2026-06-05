import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def post_prediction(text: str) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={"text": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_history(limit: int = 20) -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_BASE_URL}/history",
        params={"limit": limit},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Smart Support Router", layout="centered")

st.title("Smart Support Router")
st.caption(f"API: {API_BASE_URL}")

text = st.text_area(
    "Customer message",
    placeholder="I cannot withdraw cash from the ATM",
    height=140,
)

predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    if not text.strip():
        st.warning("Enter a customer message first.")
    else:
        try:
            with st.spinner("Predicting intent..."):
                result = post_prediction(text.strip())

            st.subheader("Prediction")
            st.metric("Intent", result["intent"])
            st.metric("Confidence", f"{result['confidence']:.2%}")

            st.subheader("Suggested Reply")
            st.write(result["suggested_reply"])

            st.subheader("Top Intents")
            st.dataframe(
                [
                    {
                        "intent": item["intent"],
                        "confidence": f"{item['confidence']:.2%}",
                    }
                    for item in result.get("top_intents", [])
                ],
                use_container_width=True,
                hide_index=True,
            )
        except requests.RequestException as exc:
            st.error(f"Prediction API is unavailable: {exc}")

st.divider()
st.subheader("Recent Predictions")

try:
    history = get_history()
    if history:
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("No prediction history yet.")
except requests.RequestException as exc:
    st.warning(f"Prediction history is unavailable: {exc}")
