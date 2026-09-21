import os
from langchain_community.vectorstores import FAISS

INDEX_PATH = "vectorstore/faiss_index"

def save_vector_store(vector_store):
    vector_store.save_local(INDEX_PATH)

def load_vector_store(embeddings):
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

def index_exists():
    return os.path.exists(INDEX_PATH)