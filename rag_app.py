import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import rag_backend as backend

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .api-key-note {
        font-size: 0.75rem;
        color: #888;
        margin-top: -8px;
        margin-bottom: 8px;
    }
    .status-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .pill-ok   { background: #d4edda; color: #155724; }
    .pill-warn { background: #fff3cd; color: #856404; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session-state bootstrap
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "message_history": [],
        "uploaded_file_name": None,
        "llm_ready": False,
        "groq_api_key": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ---------------------------------------------------------------------------
# Sidebar — API Key
# ---------------------------------------------------------------------------
st.sidebar.title("🧠 DocuMind")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Groq API Key")

api_key_input = st.sidebar.text_input(
    label="Enter your Groq API key",
    type="password",
    placeholder="gsk_…",
    value=st.session_state["groq_api_key"],
    help="Your key is kept in session memory only — never stored or logged.",
)
st.sidebar.markdown(
    "<p class='api-key-note'>Your key is used only for this session.</p>",
    unsafe_allow_html=True,
)

if api_key_input and api_key_input != st.session_state["groq_api_key"]:
    st.session_state["groq_api_key"] = api_key_input
    with st.sidebar:
        with st.spinner("Connecting to Groq…"):
            ok, msg = backend.init_llm(api_key_input)
    if ok:
        st.session_state["llm_ready"] = True
        st.sidebar.success(" API key accepted")
    else:
        st.session_state["llm_ready"] = False
        st.sidebar.error(f" {msg}")

elif api_key_input and not st.session_state["llm_ready"]:
    ok, msg = backend.init_llm(api_key_input)
    if ok:
        st.session_state["llm_ready"] = True

# ---------------------------------------------------------------------------
# Sidebar — New Chat button
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
if st.sidebar.button(" New Chat", use_container_width=True):
    st.session_state["message_history"] = []

# ---------------------------------------------------------------------------
# Sidebar — PDF Upload
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    label="Upload a PDF document",
    type=["pdf"],
    help="The chatbot will answer questions based on the content of this PDF.",
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state["uploaded_file_name"]:
        with st.sidebar.status("Processing PDF…", expanded=False):
            ok, msg = backend.load_pdf(uploaded_file.read(), uploaded_file.name)
        if ok:
            st.session_state["uploaded_file_name"] = uploaded_file.name
            st.sidebar.success(f" Ready: `{uploaded_file.name}`")
        else:
            st.sidebar.error(f" {msg}")
    else:
        st.sidebar.success(f" Ready: `{uploaded_file.name}`")
else:
    if not backend.rag_ready:
        st.sidebar.info("Upload a PDF to enable document Q&A.")

# ---------------------------------------------------------------------------
# Main area — Status bar
# ---------------------------------------------------------------------------
col_llm, col_pdf = st.columns([1, 1])
with col_llm:
    if st.session_state["llm_ready"]:
        st.markdown("<span class='status-pill pill-ok'>🟢 LLM connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-pill pill-warn'>🟡 Enter Groq API key</span>", unsafe_allow_html=True)
with col_pdf:
    if backend.rag_ready:
        st.markdown(f"<span class='status-pill pill-ok'>📄 {backend.loaded_filename}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-pill pill-warn'>📂 No PDF loaded</span>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main area — Welcome screen
# ---------------------------------------------------------------------------
if not st.session_state["message_history"]:
    st.markdown(
        """
        <div style='text-align:center; margin-top:120px; color:#888;'>
            <div style='font-size:3rem;'>🧠</div>
            <h2 style='font-weight:600; margin-bottom:8px;'>DocuMind</h2>
            <p style='font-size:1rem;'>Enter your Groq API key → Upload a PDF → Start asking questions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main area — Chat history
# ---------------------------------------------------------------------------
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Main area — Chat input
# ---------------------------------------------------------------------------
ready = st.session_state["llm_ready"]

placeholder = (
    "Ask a question about your document…"
    if ready
    else "Enter your Groq API key in the sidebar to start chatting."
)

user_input = st.chat_input(placeholder, disabled=not ready)

if user_input and ready:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        def _stream_response():
            for chunk, _meta in backend.chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                stream_mode="messages",
            ):
                if (
                    isinstance(chunk, AIMessage)
                    and chunk.content
                    and not getattr(chunk, "tool_calls", None)
                ):
                    yield chunk.content

        ai_response = st.write_stream(_stream_response())

    if ai_response:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_response}
        )
