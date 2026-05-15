from __future__ import annotations

import os
import tempfile
import logging
from typing import Annotated, Optional, Tuple

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from typing import TypedDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embeddings (free, local — no API key needed)
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Global mutable state
# ---------------------------------------------------------------------------
retriever: Optional[object] = None
rag_ready: bool = False
loaded_filename: Optional[str] = None

_current_api_key: Optional[str] = None
chatbot: Optional[object] = None

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = SystemMessage(content="""You are a helpful assistant with access to a document retrieval tool called rag_tool.

STRICT RULES — follow these without exception:

1. For ANY user question, you MUST call rag_tool first. Never answer from your own knowledge.

2. After receiving the tool result, check carefully:
   - If the tool returns useful content, answer the question based only on that content.
   - If the tool says no PDF is uploaded, politely ask the user to upload one.
   - If the tool returns content but it does not contain a relevant answer to the question,
     respond with something like: "I couldn't find an answer to that in the uploaded document."
     Do NOT make up an answer or pull from general knowledge.

3. Never expose raw tool output, JSON, metadata, file paths, or page numbers in your response.

4. Keep answers clear, concise, and grounded strictly in the document content.""")


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def _build_graph(api_key: str):
    lm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    lm_forced = lm.bind_tools([rag_tool], tool_choice="any")
    lm_free = lm.bind_tools([rag_tool])

    def chat_node(state: ChatState):
        from langchain_core.messages import ToolMessage
        messages = state["messages"]
        last_is_tool_result = bool(messages) and isinstance(messages[-1], ToolMessage)
        active_lm = lm_free if last_is_tool_result else lm_forced
        response = active_lm.invoke([SYSTEM_PROMPT] + messages)
        return {"messages": [response]}

    g = StateGraph(ChatState)
    g.add_node("chat_node", chat_node)
    g.add_node("tools", ToolNode([rag_tool]))
    g.add_edge(START, "chat_node")
    g.add_conditional_edges("chat_node", tools_condition)
    g.add_edge("tools", "chat_node")

    return g.compile()


# ---------------------------------------------------------------------------
# Public: initialise LLM
# ---------------------------------------------------------------------------
def init_llm(api_key: str) -> Tuple[bool, str]:
    global _current_api_key, chatbot

    if not api_key or not api_key.strip():
        return False, "API key must not be empty."

    api_key = api_key.strip()

    if api_key == _current_api_key and chatbot is not None:
        return True, "Already initialised."

    try:
        chatbot = _build_graph(api_key)
        _current_api_key = api_key
        logger.info("LLM graph initialised successfully.")
        return True, "LLM initialised successfully."
    except Exception as exc:
        logger.exception("Failed to initialise LLM.")
        chatbot = None
        _current_api_key = None
        return False, f"Failed to initialise LLM: {exc}"


# ---------------------------------------------------------------------------
# Public: load PDF
# ---------------------------------------------------------------------------
def load_pdf(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
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
# RAG Tool
# ---------------------------------------------------------------------------
@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the uploaded PDF document.
    Always call this tool before answering any user question.
    Returns retrieved text passages, or a message explaining why retrieval failed.
    """
    if not rag_ready or retriever is None:
        return "NO_PDF: No PDF document has been uploaded yet."

    try:
        results = retriever.invoke(query)

        if not results:
            return "NO_MATCH: The document was searched but no relevant content was found for this query."

        passages = "\n\n---\n\n".join(doc.page_content for doc in results)
        return f"FOUND: Retrieved content from the document:\n\n{passages}"

    except Exception as exc:
        logger.exception("rag_tool retrieval error.")
        return f"ERROR: An error occurred while retrieving information: {exc}"
