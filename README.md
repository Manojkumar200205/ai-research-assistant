---
title: AI Research Assistant
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

\# AI Research Assistant



A RAG-based (Retrieval Augmented Generation) tool that lets you upload research paper PDFs and ask natural language questions, with answers grounded in the paper's actual content and cited sources.



\## Features

\- Upload any research paper PDF

\- Ask questions in natural language

\- Get answers grounded in the paper's content (not hallucinated)

\- View retrieved source chunks for transparency



\## Tech Stack

\- PDF Parsing: pypdf

\- Chunking: LangChain RecursiveCharacterTextSplitter

\- Embeddings: sentence-transformers (all-MiniLM-L6-v2)

\- Vector Store: FAISS

\- LLM: Llama 3.3 70B via Groq API

\- Frontend: Streamlit



\## How It Works

1\. PDF is parsed and cleaned (references/noise removed)

2\. Text is split into overlapping chunks

3\. Chunks are embedded into vectors using a sentence-transformer model

4\. On a query, the most relevant chunks are retrieved via FAISS similarity search

5\. Retrieved chunks are passed to an LLM to generate a grounded answer



\## Setup



```bash

git clone <your-repo-url>

cd ai-research-assistant

python -m venv venv

venv\\Scripts\\Activate.ps1   # Windows

pip install -r requirements.txt

```



Create a `.env` file with:

