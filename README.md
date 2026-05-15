# 🧠 DocuMind — RAG Chatbot powered by LangGraph & Groq

DocuMind is a production-ready RAG chatbot that enables intelligent conversations with PDF documents using LangGraph, Groq, FAISS, and Streamlit. It performs semantic document retrieval with local embeddings, streams grounded responses in real time, and maintains persistent multi-thread chat history with SQLite.

> Upload any PDF. Ask anything. Get precise, cited answers — instantly.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1D9E75?logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-E24B4A?logo=groq&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-EF9F27)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

**DocuMind** is a production-ready Retrieval-Augmented Generation (RAG) chatbot built with:

- **LangGraph** for stateful, multi-turn conversation orchestration
- **Groq** (Llama 3.3 70B) for ultra-fast LLM inference
- **FAISS** + **HuggingFace Embeddings** for semantic document search
- **Streamlit** for an interactive, zero-config web UI

Users paste their own Groq API key, upload any PDF, and immediately start asking questions. The chatbot retrieves only the relevant passages, synthesises a clean natural-language answer, and keeps full conversation history across sessions, all with no backend setup required.

---

## 🚀 Live Demo

> Click here to try the live app

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔑 **Bring-your-own key** | Enter your Groq API key in the sidebar — no `.env` or server secrets needed |
| 📄 **PDF ingestion** | Upload any PDF; chunked, embedded, and indexed in seconds |
| 🔍 **Semantic search** | FAISS + `all-MiniLM-L6-v2` retrieves the 4 most relevant passages |
| 🤖 **Forced tool use** | LangGraph forces `rag_tool` on every user turn — no hallucinated "no file uploaded" |
| 💬 **Multi-turn memory** | SQLite-backed `SqliteSaver` persists full conversation history per thread |
| 🕒 **Conversation sidebar** | Switch between past chats; titles auto-generated from first message |
| ⚡ **Streaming responses** | Token-by-token streaming via `st.write_stream` — no waiting for full generation |
| 🛡 **Clean output** | `rag_tool` returns plain text only — zero metadata, JSON, or file paths leaked |
| 📊 **Status bar** | Live indicators for LLM connection and PDF load state |

---

## 🛠 Tools & Technologies

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit 1.45 | Chat interface, sidebar, streaming |
| **LLM** | Groq · Llama 3.3 70B | Fast inference, tool calling |
| **Orchestration** | LangGraph 1.2 | Stateful agent graph with conditional edges |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Local, free, no API key |
| **Vector Store** | FAISS | In-memory semantic search |
| **PDF Parsing** | PyPDFLoader (pypdf 5.9) | Text extraction from PDF |
| **Memory** | SQLite + `SqliteSaver` | Persistent conversation threads |
| **Env** | python-dotenv | Optional local `.env` support |

---

## 📁 Project Structure

```
documind/
│
├── rag_app.py              # Streamlit frontend — UI, sidebar, streaming
├── rag_backend.py          # Backend — LangGraph graph, RAG tool, PDF loader
│
├── requirements.txt        # Pinned dependencies
├── .gitignore              # Excludes .env, *.db, venv, __pycache__
│
├── .streamlit/
│   └── config.toml         # Streamlit server config & theme
│
└── architecture.svg        # System architecture diagram
```

---

## ⚡ Quick Start

### 1 · Clone the repo

```bash
git clone https://github.com/muqadasejaz/DocuMind-RAG-Chatbot
cd documind
```

### 2 · Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Windows users**: if `streamlit` fails to start via the `.exe`, always use:
> ```bash
> python -m streamlit run rag_app.py
> ```

### 4 · Run the app

```bash
python -m streamlit run rag_app.py
```

The app opens at `http://localhost:8501`.

### 5 · Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) → **API Keys** → **Create API Key**. It's free.

---

## 🧭 How to Use

```
1. Enter your Groq API key in the sidebar → LLM status turns 🟢
2. Upload a PDF via the sidebar uploader → PDF status shows filename
3. Type your question in the chat box
4. Read the answer — clean, sourced from the document, no raw JSON
5. Start a new chat or resume any past conversation from the sidebar
```

**Tips:**
- Ask broad questions first: *"What is this document about?"*
- Then drill down: *"Explain the section on Phase 3"*
- The chatbot uses only what's in the PDF — it won't hallucinate facts not present in the document

---

## 🏗 Architecture

The diagram below shows how a user message flows through the system:

<img width="1216" height="880" alt="DocuMind" src="https://github.com/user-attachments/assets/4a96b236-3e9b-4b5e-8672-f678fa6cc38f" />


### Flow walkthrough

```
User types a message
       │
       ▼
 Streamlit (rag_app.py)
   Sends HumanMessage to LangGraph
       │
       ▼
 LangGraph · chat_node
   lm_forced (tool_choice="any")
   → MUST call rag_tool before answering
       │
       ▼
 rag_tool (rag_backend.py)
   Checks rag_ready flag
   → FAISS similarity search (k=4 chunks)
   → Returns plain-text passages
       │
       ▼
 LangGraph · chat_node (again)
   lm_free receives tool result
   → Synthesises natural-language answer
       │
       ▼
 Streamlit streams answer token-by-token
   → Only AIMessage text chunks rendered
   → Tool-call chunks silently filtered
       │
       ▼
 SQLite (chatbot.db)
   Conversation saved to thread
```

### Why `tool_choice="any"`?

Without forced tool use, the LLM reads the system prompt instruction *"if no PDF is uploaded, ask the user to upload one"* and sometimes answers that instruction **directly** — without ever calling `rag_tool` to check. Binding with `tool_choice="any"` on human turns eliminates this shortcut: the model must call the tool, which then reports the actual state of `rag_ready`.

---

## 🖥 GUI Preview

```
┌─────────────────────────────────────────────────────────────┐
│  🧠 RAG Chatbot                                  sidebar    │
│  ─────────────────────────────────────────────────────────  │
│  🔑 Groq API Key                                            │
│  [gsk_••••••••••••••••••••••••••]                          │
│  ─────────────────────────────────────────────────────────  │
│  ➕ New Chat                                                 │
│  ─────────────────────────────────────────────────────────  │
│  📄 Upload PDF                                              │
│  [  Drop file here or Browse  ]                             │
│  ✅ Ready: `Week 6 Handouts.pdf`                            │
│  ─────────────────────────────────────────────────────────  │
│  🕒 Recent Chats                                            │
│  💬 What is this document about?                            │
│  💬 Explain the JD Generator                                │
├─────────────────────────────────────────────────────────────┤
│                      main area                              │
│  🟢 LLM connected          📄 Week 6 Handouts.pdf           │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  You: What is discussed in this document?           │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  🤖 This document covers a production AI hiring     │   │
│  │     platform architecture. It walks through ten     │   │
│  │     phases, from defining the vision to building    │   │
│  │     the MVP, and describes each agent in the        │   │
│  │     multi-agent system including the JD Generator,  │   │
│  │     Resume Parser, and Screening Interview Agent.   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  [ Type your message…                            Send ▶ ]  │
└─────────────────────────────────────────────────────────────┘
```

---

## ☁️ Deploy to Streamlit Cloud

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — DocuMind RAG Chatbot"
git remote add origin https://github.com/muqadasejaz/DocuMind-RAG-Chatbot
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your repository and set **Main file path** to `rag_app.py`
4. Click **Deploy**

> 💡 **No secrets needed** — users supply their own Groq API key via the UI.

> ⚠️ **SQLite note** — Streamlit Cloud's filesystem is ephemeral. Conversation history resets on each deployment/restart. For persistent history in production, swap `SqliteSaver` for `AsyncPostgresSaver` using a managed PostgreSQL instance (e.g. Supabase free tier).

---

## ⚙️ Configuration

### Optional `.env` (local development only)

```env
# .env — never commit this file
GROQ_API_KEY=gsk_your_key_here
```

The app will pick it up automatically on local runs. In production (Streamlit Cloud), the user provides the key via the sidebar UI.

### Streamlit theme (`.streamlit/config.toml`)

```toml
[theme]
base = "light"
primaryColor = "#4F8BF9"
```

Edit to match your preferred colour scheme before deploying.

---

## 📋 Requirements

```
streamlit==1.45.1
langchain-groq==1.1.2
langchain-community==0.3.31
langgraph==1.2.0
faiss-cpu==1.9.0.post1
sentence-transformers==3.4.1
pypdf==5.9.0
python-dotenv==1.2.2
```

---

## 🗺 Roadmap

- [ ] PostgreSQL-backed memory for persistent multi-user sessions
- [ ] Multi-PDF support (upload and switch between documents)
- [ ] Source citation with page numbers in the response
- [ ] OpenAI / Anthropic model toggle
- [ ] Docker container for self-hosted deployment

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 👤 Author

Muqadas Ejaz

BS Computer Science (AI Specialization)

AI/ML Engineer

Kaggle Grand Master

Data Science & Gen AI 

📫 Connect with me on [LinkedIn](https://www.linkedin.com/in/muqadasejaz/)  

🌐 GitHub: [github.com/muqadasejaz](https://github.com/muqadasejaz)

📬 Kaggle: [Kaggle Profile](https://www.kaggle.com/muqaddasejaz) 


----

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  ⭐ If you find this project useful, don’t forget to star the repository!❤️
</div>
