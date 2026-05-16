# 🧠 DocuMind: RAG Chatbot powered by LangGraph & Groq

DocuMind is a RAG chatbot that lets you have intelligent conversations with any PDF document. Upload a file, ask questions, and get answers grounded strictly in the document content — no hallucinations, no made-up facts.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-1D9E75?logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-E24B4A?logo=groq&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-EF9F27)
![License](https://img.shields.io/badge/License-MIT-green)


<img width="1408" height="768" alt="rag_chatbot_langgraph" src="https://github.com/user-attachments/assets/273ae469-cba5-4b60-b7d7-28025dba3751" />


---

## 📌 Overview

**DocuMind** is a Retrieval-Augmented Generation (RAG) chatbot built with:

- **LangGraph** for stateful, multi-turn conversation orchestration
- **Groq** (Llama 3.3 70B) for fast LLM inference
- **FAISS** + **HuggingFace Embeddings** for semantic document search
- **Streamlit** for an interactive, zero-config web UI

Paste your Groq API key, upload any PDF, and start asking questions. The chatbot retrieves the most relevant passages from your document and synthesises a clean answer. If the answer is not in the document, it tells you that directly rather than guessing.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-chatbot-langgraph.streamlit.app)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔑 **Bring-your-own key** | Enter your Groq API key in the sidebar — no `.env` or server secrets needed |
| 📄 **PDF ingestion** | Upload any PDF; chunked, embedded, and indexed in seconds |
| 🔍 **Semantic search** | FAISS + `all-MiniLM-L6-v2` retrieves the 4 most relevant passages |
| 🤖 **Forced tool use** | LangGraph forces `rag_tool` on every user turn — no hallucinated answers |
| 🚫 **Honest "not found"** | If the answer is not in the document, the bot says so clearly instead of making something up |
| ⚡ **Streaming responses** | Token-by-token streaming via `st.write_stream` — no waiting for full generation |
| 🛡 **Clean output** | Zero metadata, JSON, or file paths leaked in responses |
| 📊 **Status bar** | Live indicators for LLM connection and PDF load state |

---

## 🛠 Tools & Technologies

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit 1.45 | Chat interface, sidebar, streaming |
| **LLM** | Groq · Llama 3.3 70B | Fast inference, tool calling |
| **Orchestration** | LangGraph 1.2 | Stateful agent graph with conditional edges |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Local, free, no API key needed |
| **Vector Store** | FAISS | In-memory semantic search |
| **PDF Parsing** | PyPDFLoader (pypdf) | Text extraction from PDF |

---

## 📁 Project Structure

```
├──  .streamlit/
│   └── config.toml             
├── .gitignore 
├── readme.md        
├── rag_app.py
├── rag_backend.py
└── requirements.txt    
```

---

## ⚡ Quick Start

### 1 · Clone the repo

```bash
git clone https://github.com/muqadasejaz/DocuMind-RAG-Chatbot
cd DocuMind-RAG-Chatbot
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

### 4 · Run the app

```bash
streamlit run rag_app.py
```

The app opens at `http://localhost:8501`.

### 5 · Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) then go to **API Keys** and click **Create API Key**. It is free.


---

## 🧭 How to Use

```
1. Enter your Groq API key in the sidebar  →  LLM status turns 🟢
2. Upload a PDF via the sidebar uploader   →  PDF status shows filename
3. Type your question in the chat box
4. Get a clean answer sourced strictly from the document
5. If the answer is not in the document, the bot will tell you honestly
6. Click ➕ New Chat to start a fresh conversation
```

**Tips:**
- Start broad: *"What is this document about?"*
- Then drill down: *"Explain the section on X"*
- The chatbot only uses what is in the PDF — it will not fill gaps with general knowledge

---

## 🏗 Architecture

The diagram below shows how a user message flows through the system:

<img width="1408" height="768" alt="Rag-Archi" src="https://github.com/user-attachments/assets/4b1bd2c4-8598-45eb-a92c-aaecff114731" />



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
→ Returns one of:
FOUND:     relevant passages from the document
NO_MATCH:  document searched, nothing relevant found
NO_PDF:    no document uploaded yet
ERROR:     something went wrong during retrieval
│
▼
LangGraph · chat_node (again)
lm_free receives tool result and responds:
→ FOUND     →  answers based on retrieved passages
→ NO_MATCH  →  "I couldn't find that in the document"
→ NO_PDF    →  asks user to upload a PDF first
│
▼
Streamlit streams answer token-by-token
→ Only AIMessage text chunks rendered
→ Tool-call chunks silently filtered
```

<img width="1407" height="768" alt="rag chatbot" src="https://github.com/user-attachments/assets/0db6d6e9-3e47-4f43-a79f-0aba81dcd3b3" />


### Why `tool_choice="any"`?

Without forced tool use, the LLM sometimes reads the system prompt and answers directly, skipping the retrieval step entirely. Binding with `tool_choice="any"` on every human turn forces the model to call `rag_tool` first, which then reports the actual state of the document index. No shortcuts, no guessing.


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
langchain-core>=0.3.0
langchain-groq==1.1.2
langchain-community==0.3.31
langchain-text-splitters>=0.3.0
langgraph==1.2.0
faiss-cpu>=1.12.0
sentence-transformers
pypdf
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

This project is licensed under the **MIT License**

⭐ If you find this project useful, don’t forget to star the repository!
