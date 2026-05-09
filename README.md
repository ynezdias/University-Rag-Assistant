# University-Rag-Assistant

<div align="center">

<img src="https://img.shields.io/badge/Stevens_Institute_of_Technology-RAG_Assistant-c89b3c?style=for-the-badge&logo=graduation-cap&logoColor=white" alt="Stevens RAG"/>

# 🎓 University RAG Assistant

**An AI-powered document intelligence system for Stevens Institute of Technology**  
Ask questions about admissions, tuition, deadlines, courses, and international student policies —  
and get cited, conflict-aware answers grounded strictly in official university documents.

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-6c47ff?style=flat-square)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-f55036?style=flat-square)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-28a05a?style=flat-square)](LICENSE)

<br/>

![Demo Screenshot Placeholder](https://placehold.co/860x420/0b1120/c89b3c?text=Stevens+RAG+Assistant&font=playfair-display)

</div>

---

## ✨ What It Does

This project turns five official Stevens Institute PDF documents into a **queryable knowledge base**. You ask a question in natural language; the system retrieves the most relevant chunks, sends them to an LLM, and returns a sourced, structured answer.

The standout feature is **conflict detection** — if two documents disagree on the same fact (e.g. two different application deadlines), the app doesn't silently pick one. It:

1. 🟢 Highlights the **preferred answer** from the more recently published source
2. 🔴 Flags the **conflict** with both values and their sources side-by-side
3. Recommends the student verify directly with the relevant office

---

## 📂 Project Structure

```
University-Rag-Assistant/
│
├── app.py                    # Streamlit UI — two-column layout, conflict cards
│
├── src/
│   ├── __init__.py
│   ├── rag.py                # Retrieval + LLM answering + conflict-aware prompt
│   └── ingest.py             # PDF loading, sentence-aware chunking, ChromaDB upsert
│
├── data/                     # Drop your PDF documents here
│   ├── 01_Stevens_Admissions_Guide.pdf
│   ├── 02_Stevens_Course_Catalogue.pdf
│   ├── 03_Stevens_Tuition_Fees.pdf
│   ├── 04_Stevens_Academic_Calendar.pdf
│   └── 05_Stevens_International_FAQ.pdf
│
├── chroma_db/                # Auto-created by ingest.py (git-ignored)
├── .env                      # Your API keys (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INGEST PIPELINE                         │
│                                                                 │
│  PDF files  →  PyPDF page extract  →  Sentence-aware chunker   │
│            →  all-MiniLM-L6-v2 embeddings  →  ChromaDB upsert  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                         │
│                                                                 │
│  User question  →  Embed query  →  ChromaDB top-8 retrieval    │
│               →  Build context (with chunk + page labels)       │
│               →  Conflict-aware prompt  →  Groq LLaMA 3.3 70B  │
│               →  Parse ✓ / ⚠ markers  →  Render styled cards   │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Technology | Why |
|---|---|---|
| **Vector store** | ChromaDB (persistent) | Local, fast, no infra needed |
| **Embeddings** | `all-MiniLM-L6-v2` | 384-d semantic vectors, free |
| **LLM** | LLaMA 3.3 70B via Groq | Strong multi-source reasoning, free tier |
| **Chunking** | Sentence-boundary aware | Dates & deadlines never split mid-sentence |
| **UI** | Streamlit + custom CSS | Two-column, dark academic theme |

---

## 🚀 Getting Started

### 1 — Clone the repo

```bash
git clone https://github.com/yourusername/University-Rag-Assistant.git
cd University-Rag-Assistant
```

### 2 — Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Add your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 5 — Add your PDFs

Drop your university PDF documents into the `data/` folder.

### 6 — Ingest documents

```bash
python src/ingest.py
```

This extracts text, chunks it sentence-by-sentence, generates semantic embeddings, and stores everything in ChromaDB. Re-run any time you add or update documents — `upsert` makes it safe to re-run.

### 7 — Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💡 Example Questions to Try

| Question | Expected behaviour |
|---|---|
| `What is the priority deadline for graduate master's programs?` | 🔴 **Conflict detected** — two different dates found across document sections |
| `What is the minimum TOEFL score for international students?` | 🟢 Clean answer — 79 iBT, cited from International FAQ |
| `How much is undergraduate tuition for 2025–2026?` | 🟢 Clean answer — $62,428, cited from Tuition & Fees doc |
| `When does the fall 2025 semester start?` | 🟢 Clean answer — August 28, 2025, from Academic Calendar |
| `Can F-1 students work off-campus in their first year?` | 🟢 Clean answer with policy details from International FAQ |

---

## ⚠️ Conflict Detection in Action

When two sources disagree, the UI renders two distinct cards:

```
┌─────────────────────────────────────────────────────┐
│ ✓  PREFERRED ANSWER (most recent source)            │
│                                                     │
│  Based on Source 1 (Admissions Guide, Aug 2024):    │
│  The priority deadline is February 1, 2025.         │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ ⚠  CONFLICT DETECTED                               │
│                                                     │
│  Source 3 (Admissions Guide, page 2, chunk 2)       │
│  states: January 15, 2025.                          │
│                                                     │
│  Please confirm with the Office of Admissions.      │
└─────────────────────────────────────────────────────┘
```

The LLM follows a strict 5-rule resolution protocol — it never silently picks a value when sources disagree.

---

## 🛠 Troubleshooting

**OpenBLAS memory error on Windows**
```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python src/ingest.py
```

**`src` module not found**
Make sure `src/__init__.py` exists and you're running commands from the project root, not inside `src/`.

**Conflict not detected**
Ensure `build_context()` in `rag.py` is NOT deduplicating chunks by page — two chunks from the same page can still conflict and must appear as separate sources.

**Re-ingesting after doc changes**
Just re-run `python src/ingest.py` — `upsert()` handles duplicates safely.

---

## 📦 Requirements

```txt
streamlit
chromadb
pypdf
sentence-transformers
groq
python-dotenv
```

Install all at once:
```bash
pip install streamlit chromadb pypdf sentence-transformers groq python-dotenv
```

---

## 🗺 Roadmap

- [ ] Multi-university support (swap document sets via sidebar)
- [ ] Source confidence scores displayed per chunk
- [ ] PDF viewer panel — click a source chip to see the original page
- [ ] Chat history — multi-turn conversation with memory
- [ ] Export answers as PDF report
- [ ] Admin panel — drag-and-drop document upload + re-ingest trigger

---

## 🤝 Contributing

Pull requests are welcome. For major changes please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**[⬆ Back to top](#-university-rag-assistant)**

</div>
