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


def get_history(limit: int = 10) -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_BASE_URL}/history",
        params={"limit": limit},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def format_confidence(confidence: float) -> str:
    return f"{confidence:.2%}"


def render_top_intents(top_intents: list[dict[str, Any]]) -> None:
    if not top_intents:
        return

    with st.expander("Routing details"):
        st.dataframe(
            [
                {
                    "intent": item["intent"],
                    "confidence": format_confidence(item["confidence"]),
                }
                for item in top_intents
            ],
            use_container_width=True,
            hide_index=True,
        )


def render_history() -> None:
    with st.sidebar:
        st.header("Recent requests")
        try:
            history = get_history()
        except requests.RequestException:
            st.caption("History is unavailable right now.")
            return

        if not history:
            st.caption("No recent requests yet.")
            return

        for item in history:
            st.markdown(f"**{item['predicted_intent']}**")
            st.caption(item["text"])
            st.caption(format_confidence(item["confidence"]))
            st.divider()


st.set_page_config(page_title="Smart Support Router", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi. Tell me what happened, and I will route your request to the right support team.",
        }
    ]

st.title("Smart Support")
st.caption("Customer support assistant")

render_history()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "intent" in message:
            col_intent, col_confidence = st.columns(2)
            col_intent.metric("Intent", message["intent"])
            col_confidence.metric("Confidence", format_confidence(message["confidence"]))
            render_top_intents(message.get("top_intents", []))

prompt = st.chat_input("Describe your issue")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking your request..."):
                result = post_prediction(prompt)
        except requests.RequestException:
            content = "Sorry, support routing is unavailable right now. Please try again later."
            st.write(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
        else:
            content = result["suggested_reply"]
            st.write(content)

            col_intent, col_confidence = st.columns(2)
            col_intent.metric("Intent", result["intent"])
            col_confidence.metric("Confidence", format_confidence(result["confidence"]))
            render_top_intents(result.get("top_intents", []))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "intent": result["intent"],
                    "confidence": result["confidence"],
                    "top_intents": result.get("top_intents", []),
                }
            )
