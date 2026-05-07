import os
from dotenv import load_dotenv

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI


load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "university_docs"


def get_collection():
    openai_api_key = os.getenv("OPENAI_API_KEY")

    embedding_function = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small"
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    return collection


def retrieve_chunks(question: str, top_k: int = 5):
    collection = get_collection()

    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, metadata, distance in zip(documents, metadatas, distances):
        chunks.append({
            "text": doc,
            "metadata": metadata,
            "distance": distance
        })

    return chunks


def build_context(chunks):
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]

        source_label = (
            f"Source {i}: "
            f"{metadata['filename']}, "
            f"page {metadata['page_number']}"
        )

        context_parts.append(
            f"{source_label}\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(question: str, chunks):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = build_context(chunks)

    prompt = f"""
You are a University RAG Assistant.

Answer the student's question ONLY using the provided context.

Rules:
1. If the answer is not in the context, say:
   "I don't know based on the provided university documents."
2. Do not make up deadlines, fees, requirements, policies, or course details.
3. Keep the answer clear and student-friendly.
4. Mention the source document/page in the answer when useful.

Question:
{question}

Context:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You answer only from provided university documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


def ask_university_bot(question: str, top_k: int = 5):
    chunks = retrieve_chunks(question, top_k=top_k)
    answer = generate_answer(question, chunks)

    sources = []

    for chunk in chunks:
        metadata = chunk["metadata"]
        sources.append({
            "filename": metadata["filename"],
            "page_number": metadata["page_number"],
            "chunk_number": metadata["chunk_number"]
        })

    return answer, sources, chunks