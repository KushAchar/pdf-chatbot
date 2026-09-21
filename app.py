from dotenv import load_dotenv
load_dotenv()

from utils.embeddings import get_embeddings
from utils.loader import load_pdf
from utils.splitter import split_documents
from langchain_community.vectorstores import FAISS
from utils.chat_model import get_llm
from utils.prompt import prompt

from utils.indexer import (
    save_vector_store,
    load_vector_store,
    index_exists
)

embeddings = get_embeddings()

if index_exists():
    print("Loading existing vector database...")

    vector_store = load_vector_store(embeddings)

else:
    print("Creating new vector database...")

    documents = load_pdf("data/DataCenter.pdf")
    chunks = split_documents(documents)

    print("Number of documents:", len(documents))
    print("Number of chunks:", len(chunks))

    vector_store = FAISS.from_documents(
        chunks,
        embeddings

    )

    save_vector_store(vector_store)

    print("Vector database created and saved successfully!")

# Create LLM
llm = get_llm()

# User question
query = "What is the use of a Data Center?"

# Retrieve relevant chunks
results = vector_store.similarity_search(query, k=3)

# Combine retrieved chunks
context = "\n\n".join(
    doc.page_content
    for doc in results
)

# Create prompt
formatted_prompt = prompt.invoke(
    {
        "context": context,
        "question": query
    }
)

# Ask Gemini
response = llm.invoke(formatted_prompt)

print("\nAnswer:")
print(response.content)