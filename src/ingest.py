import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "university_docs"


def load_pdf_pages(file_path: Path) -> List[Dict]:
    """
    Extract text page by page from a PDF.
    Each page becomes a raw document before chunking.
    """
    reader = PdfReader(str(file_path))
    pages = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "text": text,
                "filename": file_path.name,
                "page_number": page_index + 1
            })

    return pages


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple character-based chunking.
    Good enough for version 1.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def ingest_documents():
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    embedding_function = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small"
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    all_ids = []
    all_documents = []
    all_metadatas = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/ folder.")
        return

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        pages = load_pdf_pages(pdf_file)

        for page in pages:
            chunks = split_text(page["text"])

            for chunk_index, chunk in enumerate(chunks):
                chunk_id = f"{page['filename']}_page_{page['page_number']}_chunk_{chunk_index}"

                metadata = {
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "chunk_number": chunk_index
                }

                all_ids.append(chunk_id)
                all_documents.append(chunk)
                all_metadatas.append(metadata)

    if all_documents:
        collection.add(
            ids=all_ids,
            documents=all_documents,
            metadatas=all_metadatas
        )

    print(f"Ingestion complete. Added {len(all_documents)} chunks.")


if __name__ == "__main__":
    ingest_documents()