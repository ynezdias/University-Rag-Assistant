"""Corpus isolation and atomic index publication."""
import json
from pathlib import Path
from src.embeddings import MODEL_NAME

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
MANIFEST = CHROMA_DIR / "quackquery.json"
CORPORA = ("synthetic", "unverified")


def read_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def active_collection(corpus="synthetic"):
    if corpus not in CORPORA:
        raise ValueError("Unknown corpus")
    entry = read_manifest().get(corpus)
    if not entry:
        raise RuntimeError(f"Index not ready. Run: python -m src.ingest --corpus {corpus}")
    if entry["embedding_model"] != MODEL_NAME:
        raise RuntimeError("Embedding configuration changed. Re-ingest this corpus.")
    return entry["collection"]
