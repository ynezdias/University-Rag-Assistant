import chromadb
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "university_docs"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def retrieve_chunks(question, top_k=5):
    collection = get_collection()

    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    chunks = []

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    for doc, meta in zip(docs, metas):
        chunks.append({
            "text": doc,
            "metadata": meta
        })

    return chunks


def build_context(chunks):
    context = ""

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]

        context += f"""
Source {i}
File: {meta['filename']}
Page: {meta['page_number']}

{chunk['text']}

---
"""

    return context


def generate_answer(question, chunks):
    context = build_context(chunks)

    prompt = f"""
You are a University RAG Assistant.

Answer ONLY using the context below.

If the answer is not present, say:
"I don't know based on the university documents."

Question:
{question}

Context:
{context}
"""

    response = ollama.chat(
        model="gemma:2b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


def ask_university_bot(question, top_k=5):
    chunks = retrieve_chunks(question, top_k)
    answer = generate_answer(question, chunks)

    return answer, chunks