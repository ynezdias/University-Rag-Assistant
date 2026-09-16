"""QuackQuery corpus ingestion. Run with python -m src.ingest."""
import argparse
import hashlib
import json
import re
import sys
import uuid
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.embeddings import MODEL_NAME, embed_batch
from src.knowledge import DATA_DIR, CHROMA_DIR, MANIFEST, CORPORA, read_manifest

CHUNK_SIZE, OVERLAP, MIN_CHUNK = 900, 150, 1
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def word_text(element):
    parts = []
    for node in element.iter():
        if node.tag == W + "t":
            parts.append(node.text or "")
        elif node.tag in (W + "br", W + "cr"):
            parts.append("\n")
        elif node.tag == W + "tab":
            parts.append("\t")
        elif node.tag == W + "p" and parts:
            parts.append("\n")
    return "".join(parts)


def load_document(path):
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return [(f"Page {i}", p.extract_text() or "")
                for i, p in enumerate(PdfReader(str(path)).pages, 1)]
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    blocks = []
    for block in root.find(W + "body"):
        if block.tag == W + "p":
            blocks.append(word_text(block))
        elif block.tag == W + "tbl":
            for row in block.findall(W + "tr"):
                blocks.append(" | ".join(word_text(cell)
                                         for cell in row.findall(W + "tc")))
    return [("Document text", "\n".join(blocks))]


def discover_documents(corpus, data_dir=DATA_DIR):
    if corpus not in CORPORA:
        raise ValueError("Unknown corpus")
    files = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in (".pdf", ".docx") or path.name.startswith("~$"):
            continue
        synthetic = "synthetic" in path.relative_to(data_dir).as_posix().lower()
        if synthetic != (corpus == "synthetic"):
            continue
        if path.suffix.lower() == ".docx" and path.with_suffix(".pdf").exists():
            continue
        files.append(path)
    return files


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


def prepare_corpus(corpus, data_dir=DATA_DIR):
    ids, documents, metadatas, inventory = [], [], [], []
    for path in discover_documents(corpus, data_dir):
        sections = load_document(path)
        full_text = "\n".join(text for _, text in sections)
        if not full_text.strip():
            raise ValueError(f"No text extracted: {path.name}; existing index is unchanged.")
        relative = path.relative_to(data_dir).as_posix()
        published = re.search(r"Publication date:\s*([^\n]+)", full_text)
        base = {"filename": path.name, "document_id": relative,
                "content_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_type": "synthetic_test_data" if corpus == "synthetic" else "unverified",
                "publication_date": published.group(1).strip() if published else "unknown"}
        inventory.append({**base, "characters": len(full_text)})
        for locator, text in sections:
            for i, chunk in enumerate(split_text(text)):
                ids.append(hashlib.sha256(f"{relative}:{locator}:{i}".encode()).hexdigest())
                documents.append(chunk)
                metadatas.append({**base, "locator": locator, "chunk_number": i})
    return ids, documents, metadatas, inventory


def ingest_documents(corpus="synthetic", dry_run=False):
    ids, documents, metadatas, inventory = prepare_corpus(corpus)
    print(f"{corpus}: {len(inventory)} documents, {len(documents)} chunks")
    if dry_run:
        return inventory
    if not documents:
        raise ValueError("No documents found; existing index is unchanged.")
    import chromadb
    vectors = embed_batch(documents)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = f"quackquery_{corpus}_{uuid.uuid4().hex}"
    collection = client.create_collection(name=name, metadata={"embedding_model": MODEL_NAME})
    for start in range(0, len(ids), 500):
        end = start + 500
        collection.upsert(ids=ids[start:end], documents=documents[start:end],
                          metadatas=metadatas[start:end], embeddings=vectors[start:end])
    manifest = read_manifest()
    manifest[corpus] = {"collection": name, "embedding_model": MODEL_NAME,
                        "chunks": len(ids), "documents": inventory}
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(f"Published {name}. Previous snapshots retained for recovery.")
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=CORPORA, default="synthetic")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ingest_documents(args.corpus, args.dry_run)
