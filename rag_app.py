"""
rag_app.py  —  Streamlit frontend for the RAG Chatbot
======================================================
Changes from original:
  • Groq API key is entered via a sidebar text box — no .env required.
  • LLM is initialised lazily on first key submission (or key change).
  • Metadata / raw tool output is never shown — fixed in rag_backend.py.
  • Proper streaming filter: only final AI text chunks are rendered.
  • Guard rails: chat input disabled until both API key and PDF are ready.
  • Conversation sidebar shows proper titles; "New Chat" entries are hidden.
  • All state properly scoped to st.session_state.
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import rag_backend as backend

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Sidebar key input */
    .api-key-note {
        font-size: 0.75rem;
        color: #888;
        margin-top: -8px;
        margin-bottom: 8px;
    }
    /* Status pill */
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
        "thread_id": str(uuid.uuid4()),
        "chat_threads": [],          # populated after LLM init
        "uploaded_file_name": None,
        "llm_ready": False,
        "groq_api_key": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _new_chat():
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["message_history"] = []
    # Register the new thread
    if st.session_state["thread_id"] not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(st.session_state["thread_id"])


def _load_conversation(thread_id: str) -> list[dict]:
    """Load a past conversation as a list of {role, content} dicts."""
    try:
        state = backend.chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        raw = state.values.get("messages", [])
        result = []
        for msg in raw:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                result.append({"role": "assistant", "content": msg.content})
        return result
    except Exception:
        return []


def _thread_title(thread_id: str) -> str:
    """Use the first user message of a thread as its sidebar title."""
    try:
        state = backend.chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        for msg in state.values.get("messages", []):
            if isinstance(msg, HumanMessage):
                text = msg.content.strip()
                return (text[:38] + "…") if len(text) > 38 else text
    except Exception:
        pass
    return ""


def _register_current_thread():
    tid = st.session_state["thread_id"]
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)


# ---------------------------------------------------------------------------
# Sidebar — API Key
# ---------------------------------------------------------------------------
st.sidebar.title("🧠 RAG Chatbot")
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

# Initialise / re-initialise LLM when the key changes
if api_key_input and api_key_input != st.session_state["groq_api_key"]:
    st.session_state["groq_api_key"] = api_key_input
    with st.sidebar:
        with st.spinner("Connecting to Groq…"):
            ok, msg = backend.init_llm(api_key_input)
    if ok:
        st.session_state["llm_ready"] = True
        # Pull any existing threads from the DB now that the graph is alive
        st.session_state["chat_threads"] = backend.retrieve_threads()
        _register_current_thread()
        st.sidebar.success("✅ API key accepted")
    else:
        st.session_state["llm_ready"] = False
        st.sidebar.error(f"❌ {msg}")

elif api_key_input and not st.session_state["llm_ready"]:
    # Page rerun with same key but LLM not yet initialised (e.g. first load)
    ok, msg = backend.init_llm(api_key_input)
    if ok:
        st.session_state["llm_ready"] = True
        st.session_state["chat_threads"] = backend.retrieve_threads()
        _register_current_thread()

# ---------------------------------------------------------------------------
# Sidebar — New Chat button
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
if st.sidebar.button("➕ New Chat", use_container_width=True):
    _new_chat()

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
            st.sidebar.success(f"✅ Ready: `{uploaded_file.name}`")
        else:
            st.sidebar.error(f"❌ {msg}")
    else:
        st.sidebar.success(f"✅ Ready: `{uploaded_file.name}`")
else:
    if not backend.rag_ready:
        st.sidebar.info("Upload a PDF to enable document Q&A.")

# ---------------------------------------------------------------------------
# Sidebar — Recent Conversations
# ---------------------------------------------------------------------------
if st.session_state["llm_ready"] and st.session_state["chat_threads"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🕒 Recent Chats")

    for tid in reversed(st.session_state["chat_threads"]):
        title = _thread_title(tid)
        if not title:
            continue  # skip empty / uninitialised threads
        btn_label = f"💬 {title}"
        if st.sidebar.button(btn_label, key=f"thread_{tid}", use_container_width=True):
            st.session_state["thread_id"] = tid
            st.session_state["message_history"] = _load_conversation(tid)

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
            <h2 style='font-weight:600; margin-bottom:8px;'>RAG Chatbot</h2>
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
# Main area — Chat input (disabled until LLM is ready)
# ---------------------------------------------------------------------------
ready = st.session_state["llm_ready"]

placeholder = (
    "Type your message…"
    if ready
    else "Enter your Groq API key in the sidebar to start chatting."
)

user_input = st.chat_input(placeholder, disabled=not ready)

if user_input and ready:
    # Show user message immediately
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # Register thread (in case it wasn't already)
    _register_current_thread()

    with st.chat_message("assistant"):
        def _stream_response():
            """
            Stream only final AIMessage text chunks.
            Skips tool-call chunks, tool results, and empty content.
            """
            for chunk, _meta in backend.chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
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
