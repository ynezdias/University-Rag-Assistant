"""
Stevens University RAG Assistant — rag.py
Fixes applied vs. original:
  1. Real sentence-transformer embeddings (all-MiniLM-L6-v2) replace MD5 hash vectors
  2. top_k raised to 8 so conflict-bearing chunks are more likely to co-appear
  3. Conflict-aware prompt: LLM is explicitly told to flag contradictions
  4. Source deduplication in context builder (same page shown once)
  5. Graceful fallback if sentence-transformers not installed
"""

import os
import re
import math
import hashlib

import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CHROMA_DIR   = "chroma_db"
COLLECTION   = "university_docs"

# ── Embedding ──────────────────────────────────────────────────────────────────
# Try real semantic embeddings first; fall back to improved hash embedder.

def _load_encoder():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim, fast, free
        print("[rag] Using sentence-transformers (semantic embeddings)")
        return model
    except ImportError:
        print("[rag] sentence-transformers not found — using hash fallback.")
        print("      Install with:  pip install sentence-transformers")
        return None

_ENCODER = _load_encoder()
EMBED_DIM = 384


def embed(text: str) -> list[float]:
    """Return a normalised 384-d embedding for text."""
    if _ENCODER is not None:
        vec = _ENCODER.encode(text, normalize_embeddings=True).tolist()
        return vec
    # ── Fallback: character-level bigram + word hashing (better than plain MD5 word hash)
    vector = [0.0] * EMBED_DIM
    text_l = text.lower()
    # word-level
    for word in re.findall(r"\b\w+\b", text_l):
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBED_DIM
        vector[idx] += 1.0
    # bigram-level (adds some positional signal)
    for a, b in zip(text_l, text_l[1:]):
        bigram = a + b
        idx = int(hashlib.sha1(bigram.encode()).hexdigest(), 16) % EMBED_DIM
        vector[idx] += 0.5
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else vector


# ── ChromaDB helpers ───────────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION)


def retrieve_chunks(question: str, top_k: int = 8) -> list[dict]:
    """
    Retrieve top_k chunks most relevant to *question*.
    top_k defaults to 8 (was 5) so overlapping/conflicting chunks
    from different document sections are more likely to both appear.
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=[embed(question)],
        n_results=top_k,
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({"text": doc, "metadata": meta})
    return chunks


# ── Context builder ────────────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """
    Build a numbered context string.
    Deduplicates by (filename, page_number) so the same page
    doesn't crowd out other sources.
    """
    seen   = set()
    unique = []
    for chunk in chunks:
        m   = chunk["metadata"]
        key = (m.get("filename", ""), m.get("page_number", ""))
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    parts = []
    for i, chunk in enumerate(unique, 1):
        m = chunk["metadata"]
        parts.append(
            f"[Source {i}]\n"
            f"File: {m.get('filename', 'unknown')}\n"
            f"Page: {m.get('page_number', '?')}\n\n"
            f"{chunk['text'].strip()}\n"
        )
    return "\n---\n".join(parts)


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a University RAG Assistant for Stevens Institute of Technology.
Your job is to answer student questions using ONLY the provided source excerpts.

Rules:
1. Answer using only information found in the sources below. Cite each claim
   with its [Source N] tag (e.g. "According to [Source 2]…").

2. CONFLICT DETECTION — this is critical:
   If two or more sources give different values for the same fact (e.g. two
   different deadlines for the same event, two different tuition figures),
   DO NOT pick one silently. Instead:
   • Clearly state that a conflict exists.
   • Quote both values and their sources.
   • Advise the student to verify with the official office.
   Example output for a conflict:
     "⚠ Conflict detected: [Source 1] states the priority deadline is
      February 1, 2025, while [Source 3] lists January 15, 2025 for the
      same program. Please confirm directly with the Office of Admissions."

3. If no relevant information is present in any source, say exactly:
   "I don't know based on the university documents."

4. Keep answers concise and structured. Use bullet points where helpful.
"""


def generate_answer(question: str, chunks: list[dict]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ GROQ_API_KEY missing — add it to your .env file."

    context = build_context(chunks)
    if not context.strip():
        return "I don't know based on the university documents."

    user_message = f"""Question: {question}

Sources:
{context}
"""

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # stronger model → better conflict reasoning
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Public API ─────────────────────────────────────────────────────────────────

def ask_university_bot(question: str, top_k: int = 8) -> tuple[str, list[dict]]:
    """
    Returns (answer_text, retrieved_chunks).
    Drop-in replacement for the original function signature.
    """
    chunks = retrieve_chunks(question, top_k)
    answer = generate_answer(question, chunks)
    return answer, chunks


# ── CLI convenience ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    answer, chunks = ask_university_bot(q)
    print("\n" + "="*60)
    print("ANSWER\n" + "="*60)
    print(answer)
    print("\n" + "="*60)
    print(f"SOURCES ({len(chunks)} chunks retrieved)")
    print("="*60)
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        print(f"  [{i}] {m.get('filename','?')}  p.{m.get('page_number','?')}")