"""Streamlit chat UI for the AI Document Analyst API."""

import html
from typing import Any

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
UPLOAD_URL = f"{API_BASE}/documents"
ASK_URL = f"{API_BASE}/ask/"

st.set_page_config(
    page_title="AI Document Analyst",
    page_icon="📄",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        max-width: 820px;
    }
    .source-note {
        margin: 0.85rem 0 0.1rem;
        padding-top: 0.55rem;
        border-top: 1px solid rgba(128, 128, 128, 0.28);
        color: #6b7280;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def backend_is_up() -> bool:
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or f"Request failed ({response.status_code})"

    detail = body.get("detail", response.text)
    if isinstance(detail, list):
        return "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def source_labels(sources: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for source in sources:
        name = source.get("source") or "document"
        page = source.get("page")
        if isinstance(page, int):
            # PyPDFLoader stores a zero-based page index.
            label = f"{name} · page {page + 1}"
        else:
            label = str(name)
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def render_sources(sources: list[dict[str, Any]]) -> None:
    labels = source_labels(sources)
    if not labels:
        return
    safe = " · ".join(html.escape(label) for label in labels)
    st.markdown(
        f'<p class="source-note">Sources: {safe}</p>',
        unsafe_allow_html=True,
    )


def upload_pdf(uploaded_file) -> None:
    try:
        response = requests.post(
            UPLOAD_URL,
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf",
                )
            },
            timeout=180,
        )
    except requests.RequestException as exc:
        st.sidebar.error(f"Could not reach the backend. {exc}")
        return

    if not response.ok:
        st.sidebar.error(error_detail(response))
        return

    payload = response.json()
    st.session_state.upload_message = (
        f"Chunked {payload.get('filename', uploaded_file.name)}: "
        f"{payload.get('chunks', 0)} chunks from {payload.get('pages', 0)} pages."
    )
    st.session_state.last_upload = (uploaded_file.name, uploaded_file.size)


def ask_question(question: str) -> dict[str, Any]:
    response = requests.post(ASK_URL, json={"question": question}, timeout=60)
    if not response.ok:
        raise RuntimeError(error_detail(response))
    return response.json()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_upload" not in st.session_state:
    st.session_state.last_upload = None
if "upload_message" not in st.session_state:
    st.session_state.upload_message = None

with st.sidebar:
    st.header("Document")
    st.caption("Upload a text-based PDF. The backend extracts it, chunks it, and stores the embeddings.")
    if backend_is_up():
        st.success("Backend is running")
    else:
        st.warning("Start the API at http://127.0.0.1:8000")

    uploaded = st.file_uploader("PDF file", type=["pdf"])
    if uploaded is not None and st.session_state.last_upload != (
        uploaded.name,
        uploaded.size,
    ):
        with st.spinner("Uploading and chunking..."):
            upload_pdf(uploaded)

    if st.session_state.upload_message:
        st.success(st.session_state.upload_message)

st.title("AI Document Analyst")
st.caption("Answers are written only from the PDFs you have uploaded.")

if not st.session_state.messages:
    st.info("Upload a PDF in the sidebar, then ask a question here.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources") or [])

question = st.chat_input("Ask a question about your document")
if question:
    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Reading the relevant pages..."):
                result = ask_question(question)
        except requests.RequestException as exc:
            answer = f"Could not reach the backend. {exc}"
            sources: list[dict[str, Any]] = []
            st.error(answer)
        except RuntimeError as exc:
            answer = str(exc)
            sources = []
            st.error(answer)
        else:
            answer = result.get("answer") or ""
            sources = result.get("sources") or []
            st.markdown(answer)
            render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
