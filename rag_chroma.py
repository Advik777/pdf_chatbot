# rag_chroma.py

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import uuid

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent Chroma Client
chroma_client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",
        anonymized_telemetry=False
    )
)

collection = chroma_client.get_or_create_collection(name="pdf_collection")


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def add_pdf_to_chroma(file):
    raw_text = extract_text_from_pdf(file)
    chunks = chunk_text(raw_text)

    embeddings = embedding_model.encode(chunks)

    file_id = str(uuid.uuid4())

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"file_id": file_id, "filename": file.name} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )

    return file_id


def retrieve_context(query, top_k=3):
    query_embedding = embedding_model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    return results["documents"][0] if results["documents"] else []


def list_files():
    results = collection.get(include=["metadatas"])
    filenames = set()

    if results["metadatas"]:
        for meta in results["metadatas"]:
            filenames.add(meta["filename"])

    return list(filenames)


def delete_file(filename):
    results = collection.get(include=["metadatas"])

    ids_to_delete = []

    for idx, meta in enumerate(results["metadatas"]):
        if meta["filename"] == filename:
            ids_to_delete.append(results["ids"][idx])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)