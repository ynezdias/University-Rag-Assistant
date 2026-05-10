import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import re
import math
import hashlib

import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION = "university_docs"


# ── Embedding (must match ingest.py exactly) ───────────────────────────────────

def _load_encoder():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[rag] Using sentence-transformers (semantic embeddings)")
        return model
    except ImportError:
        print("[rag] sentence-transformers not found — using hash fallback.")
        return None

_ENCODER  = _load_encoder()
EMBED_DIM = 384


def embed(text: str) -> list[float]:
    if _ENCODER is not None:
        return _ENCODER.encode(text, normalize_embeddings=True).tolist()
    vector = [0.0] * EMBED_DIM
    text_l = text.lower()
    for word in re.findall(r"\b\w+\b", text_l):
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBED_DIM
        vector[idx] += 1.0
    for a, b in zip(text_l, text_l[1:]):
        idx = int(hashlib.sha1((a+b).encode()).hexdigest(), 16) % EMBED_DIM
        vector[idx] += 0.5
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else vector


# ── ChromaDB ───────────────────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION)


def retrieve_chunks(question: str, top_k: int = 8) -> list[dict]:
    collection = get_collection()
    results    = collection.query(
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
    No page-level deduplication — two chunks from the same page
    can carry conflicting values and must appear as separate sources.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        m = chunk["metadata"]
        parts.append(
            f"[Source {i}]\n"
            f"File: {m.get('filename', 'unknown')}\n"
            f"Page: {m.get('page_number', '?')}  |  "
            f"Chunk: {m.get('chunk_number', '?')}\n\n"
            f"{chunk['text'].strip()}\n"
        )
    return "\n---\n".join(parts)


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a University RAG Assistant for Stevens Institute of Technology.
Your job is to answer student questions using ONLY the provided source excerpts.

Each source has a File name, Page number, and Chunk number.
Some sources carry a publication date in their header (e.g. "Published: August 2024").

── CONFLICT RESOLUTION RULES (follow in this exact order) ──────────────────

RULE 1 — SINGLE SOURCE, NO CONFLICT
If only one source contains the relevant information, answer from it directly
and cite it: "According to [Source N] (filename, page X)…"

RULE 2 — MULTIPLE SOURCES AGREE
If multiple sources contain the same information and agree, answer confidently
and cite all agreeing sources.

RULE 3 — CONFLICT DETECTED, DATES AVAILABLE
If two or more sources give DIFFERENT values for the same fact AND their
publication dates are visible in the source headers:
  a. Use the value from the MORE RECENTLY PUBLISHED document as your answer.
  b. Still flag the conflict clearly so the student is aware.
  c. Format your response exactly like this:

     ✓ Based on the more recent document [Source N] (filename, published DATE):
     [your answer here]

     ⚠ Conflict detected: [Source M] (filename, page X, chunk Y) states [OTHER VALUE].
     Because [Source N] is more recent, its value is preferred — but please
     confirm with the relevant office before acting on this information.

RULE 4 — CONFLICT DETECTED, NO DATES AVAILABLE
If sources conflict but no publication dates are visible:
  a. DO NOT silently pick one value.
  b. Present BOTH values with their sources.
  c. Format your response exactly like this:

     ⚠ Conflict detected — unable to determine which source is more recent:
     • [Source N] (filename, page X, chunk Y) states: [VALUE 1]
     • [Source M] (filename, page X, chunk Y) states: [VALUE 2]
     We recommend verifying this directly with the relevant Stevens office.

RULE 5 — NO RELEVANT INFORMATION
If no source contains information relevant to the question, respond with
exactly: "I don't know based on the university documents."

── GENERAL RULES ────────────────────────────────────────────────────────────

- NEVER invent, assume, or infer information not present in the sources.
- ALWAYS cite every claim with its [Source N] tag.
- Keep answers concise. Use bullet points for lists of requirements or dates.
- When flagging a conflict, quote the exact values — do not paraphrase dates or numbers.
- Two chunks from the same file but different chunk numbers are treated as
  separate sources and may still conflict with each other.
"""


def generate_answer(question: str, chunks: list[dict]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY missing — add it to your .env file."

    context = build_context(chunks)
    if not context.strip():
        return "I don't know based on the university documents."

    user_message = f"Question: {question}\n\nSources:\n{context}"

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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
    chunks = retrieve_chunks(question, top_k)
    answer = generate_answer(question, chunks)
    return answer, chunks


# ── CLI ────────────────────────────────────────────────────────────────────────

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
        print(f"  [{i}] {m.get('filename','?')}  p.{m.get('page_number','?')}  chunk {m.get('chunk_number','?')}")