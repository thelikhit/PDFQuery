import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
UPLOAD_ENDPOINT = f"{API_URL}/api/v1/upload"
HEALTH_ENDPOINT = f"{API_URL}/health"


@st.cache_data(ttl=10)
def is_backend_alive() -> bool:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def upload_pdf(file, api_key: str) -> tuple[bool, str]:
    """Returns (success, message)"""
    if not api_key:
        return False, "Please enter your API key in the sidebar."
    try:
        headers = {"x-api-key": api_key}
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(UPLOAD_ENDPOINT, headers=headers, files=files, timeout=60)
        response.raise_for_status()
        return True, f"**{file.name}** was successfully embedded and added to the vector store."
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to the backend. Is it running?"
    except requests.exceptions.Timeout:
        return False, "Upload timed out. Try a smaller file or check the server."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        if status == 401:
            return False, "Unauthorized — check your API key."
        elif status == 415:
            return False, "Unsupported file type returned by server."
        elif status == 413:
            return False, "File is too large for the server to accept."
        else:
            return False, f"Server error {status}: {e.response.text}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


st.set_page_config(page_title="Upload PDFs")

with st.sidebar:
    st.title("Upload PDFs")
    st.caption("Add documents to the vector store for querying.")
    st.divider()
    API_KEY = st.text_input(
        "API Key",
        type="password",
        help="Enter your API key to authenticate requests.",
    )
    st.divider()
    st.caption(f"API: `{API_URL}`")
    if not API_KEY:
        st.warning("API key not set")


if not is_backend_alive():
    st.error("Backend is not reachable. Please start the API server first.")
    st.stop()


st.header("Upload PDFs to Vector Store")
st.caption("Uploaded PDFs will be chunked, embedded, and stored for document Q&A.")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    accept_multiple_files=False,
    help="Only PDF files are supported. Max size depends on your server config.",
)

if uploaded_file:
    size_kb = len(uploaded_file.getvalue()) / 1024
    st.info(
        f"**{uploaded_file.name}** — "
        f"{size_kb:.1f} KB"
    )

st.divider()

submit = st.button(
    "Upload & Embed",
    disabled=not (uploaded_file and API_KEY),
    type="primary",
    use_container_width=True,
)

if submit and uploaded_file:
    with st.spinner(f"Uploading and embedding **{uploaded_file.name}**..."):
        success, message = upload_pdf(uploaded_file, API_KEY)

    if success:
        st.success(message)
    else:
        st.error(message)