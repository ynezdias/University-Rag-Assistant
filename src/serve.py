"""Single-instance container startup; publish an index when missing or changed."""
import os
from src.ingest import prepare_corpus, ingest_documents
from src.knowledge import CORPORA, read_manifest
from src.embeddings import MODEL_NAME


def ensure_index(corpus):
    current = read_manifest().get(corpus, {})
    inventory = prepare_corpus(corpus)[3]
    if current.get("embedding_model") != MODEL_NAME or current.get("documents") != inventory:
        ingest_documents(corpus)


def main():
    corpus = os.getenv("QUACKQUERY_CORPUS", "synthetic")
    if corpus not in CORPORA:
        raise ValueError("Invalid QUACKQUERY_CORPUS")
    ensure_index(corpus)
    os.execvp("python", ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"])


if __name__ == "__main__":
    main()
