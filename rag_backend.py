from __future__ import annotations

import os
import sqlite3
import tempfile
import logging
from typing import Annotated, Optional, Tuple

from dotenv import load_dotenv  # still loaded so local dev still works

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from typing import TypedDict

load_dotenv()  # optional — values are overridden by the UI key when provided

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embeddings  (free, local — no API key needed)
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Global mutable state
# ---------------------------------------------------------------------------
retriever: Optional[object] = None
rag_ready: bool = False
loaded_filename: Optional[str] = None

# LLM / graph are rebuilt when the API key changes
_current_api_key: Optional[str] = None
llm_with_tools: Optional[object] = None
chatbot: Optional[object] = None


# ---------------------------------------------------------------------------
# LLM + Graph initialisation (lazy, key-aware)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = SystemMessage(content="""You are a helpful assistant with access to a document retrieval tool called rag_tool.

CRITICAL RULES — follow these without exception:
1. For ANY question about a document, its contents, or any topic that could be in a document,
   you MUST call rag_tool first. Do not try to answer from memory.
2. The rag_tool itself will tell you if no PDF has been uploaded — you do not need to guess.
   Never say "no document is uploaded" without calling the tool first.
3. After receiving the tool result, answer in clear, natural language.
   Never expose raw tool output, JSON, metadata, file paths, or page numbers.
4. If the tool returns that no PDF is uploaded, then politely ask the user to upload one.""")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_graph(api_key: str):
    """Compile a fresh LangGraph chatbot for the given Groq API key."""

    lm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

    # Forced: always call a tool on the first LLM turn (human message).
    # Prevents the LLM from hallucinating "no PDF uploaded" without checking.
    lm_forced = lm.bind_tools([rag_tool], tool_choice="any")

    # Free: after tool results are in, let the LLM synthesise the final answer.
    lm_free = lm.bind_tools([rag_tool])

    def chat_node(state: ChatState):
        from langchain_core.messages import ToolMessage
        messages = state["messages"]
        # Use free LLM to synthesise after tool result; force tool call otherwise.
        last_is_tool_result = bool(messages) and isinstance(messages[-1], ToolMessage)
        active_lm = lm_free if last_is_tool_result else lm_forced
        response = active_lm.invoke([SYSTEM_PROMPT] + messages)
        return {"messages": [response]}

    conn = sqlite3.connect(
        database="chatbot.db",
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(conn)

    g = StateGraph(ChatState)
    g.add_node("chat_node", chat_node)
    g.add_node("tools", ToolNode([rag_tool]))
    g.add_edge(START, "chat_node")
    g.add_conditional_edges("chat_node", tools_condition)
    g.add_edge("tools", "chat_node")

    return g.compile(checkpointer=checkpointer), checkpointer


def init_llm(api_key: str) -> Tuple[bool, str]:
    """
    Initialise (or re-initialise) the LLM + graph with the supplied key.
    Returns (success, message).
    """
    global _current_api_key, llm_with_tools, chatbot, _checkpointer

    if not api_key or not api_key.strip():
        return False, "API key must not be empty."

    api_key = api_key.strip()

    if api_key == _current_api_key and chatbot is not None:
        return True, "Already initialised."

    try:
        chatbot, _checkpointer = _build_graph(api_key)
        _current_api_key = api_key
        logger.info("LLM graph initialised successfully.")
        return True, "LLM initialised successfully."
    except Exception as exc:
        logger.exception("Failed to initialise LLM.")
        chatbot = None
        _current_api_key = None
        return False, f"Failed to initialise LLM: {exc}"


_checkpointer = None  # set by _build_graph


# ---------------------------------------------------------------------------
# PDF loader
# ---------------------------------------------------------------------------

def load_pdf(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Build a FAISS vector store from uploaded PDF bytes.
    Returns (success, message).
    """
    global retriever, rag_ready, loaded_filename

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if not docs:
            return False, "The PDF appears to be empty or could not be parsed."

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        rag_ready = True
        loaded_filename = filename
        logger.info("PDF loaded: %s (%d chunks)", filename, len(chunks))
        return True, f"Loaded {len(chunks)} chunks from '{filename}'"

    except Exception as exc:
        rag_ready = False
        logger.exception("PDF load failed.")
        return False, str(exc)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# RAG Tool  —  returns a PLAIN STRING, never a dict
# ---------------------------------------------------------------------------

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the uploaded PDF document.
    Use this tool when the user asks factual or conceptual questions
    that might be answered from the uploaded document.
    Returns the retrieved text passages as a single plain-text string.
    """
    if not rag_ready or retriever is None:
        return "No PDF document has been uploaded yet. Please ask the user to upload one."

    try:
        results = retriever.invoke(query)
        if not results:
            return "No relevant information was found in the document for that query."

        # Join context chunks with a clear separator — NO metadata exposed
        passages = "\n\n---\n\n".join(doc.page_content for doc in results)
        return f"Retrieved context from the document:\n\n{passages}"

    except Exception as exc:
        logger.exception("rag_tool retrieval error.")
        return f"An error occurred while retrieving information: {exc}"


# ---------------------------------------------------------------------------
# Thread helpers
# ---------------------------------------------------------------------------

def retrieve_threads() -> list[str]:
    """Return all known thread IDs from the checkpointer."""
    if _checkpointer is None:
        return []
    try:
        return list(
            {
                cp.config["configurable"]["thread_id"]
                for cp in _checkpointer.list(None)
            }
        )
    except Exception:
        logger.warning("Could not retrieve threads.", exc_info=True)
        return []
