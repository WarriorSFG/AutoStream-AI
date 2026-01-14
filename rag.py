import json
import os
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def setup_rag_pipeline(file_path="data.json"):
    # 1. Load the JSON data
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure it exists.")
        return None

    # 2. Convert JSON to text format
    text_data = []
    for section, content in data.items():
        text_data.append(f"Section: {section}\nDetails: {content}")

    # 3. Create Documents
    documents = [Document(page_content="\n\n".join(text_data))]

    # 4. Initialize Local Embeddings
    # This downloads a small model (approx 80MB) once and runs locally.
    print("Loading local embedding model... (this happens only once)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Create Vector Store
    vector_store = FAISS.from_documents(documents, embeddings)

    print("RAG Pipeline initialized successfully.")
    return vector_store.as_retriever()


if __name__ == "__main__":
    setup_rag_pipeline()