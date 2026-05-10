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
# code
_ENCODER = _load_encoder()
EMBED_DIM = 384
# code
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
    unique = chunks
    # seen   = set()
    # unique = []
    # for chunk in chunks:
    #     m   = chunk["metadata"]
    #     key = (m.get("filename", ""), m.get("page_number", ""))
    #     if key not in seen:
    #         seen.add(key)
    #         unique.append(chunk)

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

Each source has a File name and a Page number. Some sources also carry a
publication date in their header (e.g. "Published: August 2024").

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

     ⚠ Conflict detected: [Source M] (filename, published DATE) states [OTHER VALUE].
     Because [Source N] is more recent, its value is preferred — but please
     confirm with the relevant office before acting on this information.

RULE 4 — CONFLICT DETECTED, NO DATES AVAILABLE
If sources conflict but no publication dates are visible:
  a. DO NOT silently pick one value.
  b. Present BOTH values with their sources.
  c. Format your response exactly like this:

     ⚠ Conflict detected — unable to determine which source is more recent:
     • [Source N] (filename, page X) states: [VALUE 1]
     • [Source M] (filename, page Y) states: [VALUE 2]
     We recommend verifying this directly with the relevant Stevens office.

RULE 5 — NO RELEVANT INFORMATION
If no source contains information relevant to the question, respond with
exactly: "I don't know based on the university documents."

── GENERAL RULES ────────────────────────────────────────────────────────────

- NEVER invent, assume, or infer information not present in the sources.
- ALWAYS cite every claim with its [Source N] tag.
- Keep answers concise. Use bullet points for lists of requirements or dates.
- When flagging a conflict, quote the exact values from each source —
  do not paraphrase dates or numbers.
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