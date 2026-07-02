import os
from dotenv import load_dotenv
load_dotenv()
from utils.embeddings import get_embeddings
from utils.loader import load_pdf
from utils.splitter import split_documents
from langchain_community.vectorstores import FAISS
documents = load_pdf("data/BigData.pdf")
chunks = split_documents(documents)
embeddings = get_embeddings()
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)
print("Vector db created successfully")
print("Length of documents",len(documents))
print("Length of chunks",len(chunks))

query = "What is Big Data?"

results = vector_store.similarity_search(query, k=3)

for i, result in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("-" * 40)
    print(result.page_content)