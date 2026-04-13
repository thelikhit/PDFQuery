# python -m streamlit run src/app/ui/Chat.py

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
CHAT_ENDPOINT = f"{API_URL}/api/v1/chat"
HEALTH_ENDPOINT = f"{API_URL}/health"


st.set_page_config(page_title="DocQuery", page_icon="📄")


@st.cache_data(ttl=10)
def is_backend_alive() -> bool:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


if not is_backend_alive():
    st.error("Backend is not reachable. Please start the API server first.")
    st.stop()


def query_api(question: str, API_KEY: str) -> str:
    try:
        headers = {"x-api-key": API_KEY}
        response = requests.post(
            CHAT_ENDPOINT,
            headers=headers,
            json={"prompt": question},
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("response", "No answer returned.")
    except requests.exceptions.ConnectionError:
        return "Could not connect to the backend."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"Server error {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"Unexpected error: {e}"


with st.sidebar:
    st.title("DocQuery")
    st.caption("Ask questions about your documents.")
    st.divider()
    API_KEY = st.text_input(
        "API Key",
        type="password",
        placeholder="",
        help="Enter your API key to authenticate requests.",
    )
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(f"API: `{API_URL}`")


st.header("DocQuery")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me something about your documents...", disabled=not API_KEY):
    if not API_KEY:
        st.warning("Please enter your API key in the sidebar before chatting.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = query_api(prompt, API_KEY)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})