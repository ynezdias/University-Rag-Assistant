import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import re
import math
import hashlib
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
import chromadb

DATA_DIR        = Path("data")
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "university_docs"

CHUNK_SIZE  = 900
OVERLAP     = 150
MIN_CHUNK   = 60


# ── Embedding ──────────────────────────────────────────────────────────────────

def _load_encoder():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[ingest] Using sentence-transformers (semantic embeddings)")
        return model
    except ImportError:
        print("[ingest] sentence-transformers not found — using hash fallback.")
        print("         Install with:  pip install sentence-transformers")
        return None

_ENCODER  = _load_encoder()
EMBED_DIM = 384


def embed(text: str) -> List[float]:
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


def embed_batch(texts: List[str]) -> List[List[float]]:
    if _ENCODER is not None:
        return _ENCODER.encode(texts, normalize_embeddings=True, batch_size=8).tolist()
    return [embed(t) for t in texts]


# ── PDF loading ────────────────────────────────────────────────────────────────

def load_pdf_pages(file_path: Path) -> List[Dict]:
    reader = PdfReader(str(file_path))
    pages  = []
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "text":        text,
                "filename":    file_path.name,
                "page_number": page_index + 1,
            })
    return pages


# ── Sentence-aware chunking ────────────────────────────────────────────────────

def _sentence_split(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    sentences   = _sentence_split(text)
    chunks      = []
    current     = []
    current_len = 0

    for sentence in sentences:
        slen = len(sentence)

        if current_len + slen > chunk_size and current:
            chunk_text = " ".join(current).strip()
            if len(chunk_text) >= MIN_CHUNK:
                chunks.append(chunk_text)

            carry     = []
            carry_len = 0
            for s in reversed(current):
                if carry_len + len(s) <= overlap:
                    carry.insert(0, s)
                    carry_len += len(s)
                else:
                    break
            current     = carry
            current_len = carry_len

        current.append(sentence)
        current_len += slen

    if current:
        chunk_text = " ".join(current).strip()
        if len(chunk_text) >= MIN_CHUNK:
            chunks.append(chunk_text)

    return chunks


# ── Ingest ─────────────────────────────────────────────────────────────────────

def ingest_documents():
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in '{DATA_DIR}/'. Add your PDFs there and re-run.")
        return

    all_ids, all_docs, all_metas = [], [], []

    for pdf_file in pdf_files:
        print(f"  Processing {pdf_file.name} ...")
        pages = load_pdf_pages(pdf_file)

        for page in pages:
            chunks = split_text(page["text"])
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{page['filename']}_p{page['page_number']}_c{chunk_idx}"
                all_ids.append(chunk_id)
                all_docs.append(chunk)
                all_metas.append({
                    "filename":     page["filename"],
                    "page_number":  page["page_number"],
                    "chunk_number": chunk_idx,
                })

    if not all_docs:
        print("No text extracted. Check that your PDFs are not scanned images.")
        return

    print(f"\nEmbedding {len(all_docs)} chunks ...")
    all_embeds = embed_batch(all_docs)

    BATCH = 500
    for start in range(0, len(all_ids), BATCH):
        sl = slice(start, start + BATCH)
        collection.upsert(
            ids=all_ids[sl],
            documents=all_docs[sl],
            metadatas=all_metas[sl],
            embeddings=all_embeds[sl],
        )
        print(f"  Upserted {min(start + BATCH, len(all_ids))}/{len(all_docs)} chunks")
# code
    print(f"\nDone. {len(all_docs)} chunks stored in '{CHROMA_DIR}/{COLLECTION_NAME}'")


if __name__ == "__main__":
    ingest_documents()