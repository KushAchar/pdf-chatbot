import os
import streamlit as st
from dotenv import load_dotenv

from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.embeddings import get_embeddings
from utils.indexer import (
    save_vector_store,
    load_vector_store,
    index_exists
)
from utils.chat_model import get_llm
from utils.prompt import prompt
from langchain_community.vectorstores import FAISS

load_dotenv()

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚"
)

st.title("📚 AI PDF Chatbot")
st.write("Upload a PDF and ask questions about it.")

# -----------------------------
# Upload PDF
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

# -----------------------------
# Process PDF
# -----------------------------

if uploaded_file:

    os.makedirs("data", exist_ok=True)

    pdf_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    # Create embeddings
    embeddings = get_embeddings()

    # Create / load vector store
    if index_exists():

        st.info("Loading existing vector database...")

        vector_store = load_vector_store(
            embeddings
        )

    else:

        st.info("Creating vector database...")

        documents = load_pdf(pdf_path)

        chunks = split_documents(documents)

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        save_vector_store(vector_store)

        st.success("Vector database created!")

    # -----------------------------
    # Ask Question
    # -----------------------------

    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if st.button("Ask"):

        if question:

            # Retrieve relevant chunks
            results = vector_store.similarity_search(
                question,
                k=3
            )

            # Create context
            context = "\n\n".join(
                doc.page_content
                for doc in results
            )

            # Create prompt
            formatted_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

            # Get Gemini
            llm = get_llm()

            response = llm.invoke(
                formatted_prompt
            )

            # Display answer
            st.subheader("Answer")

            st.write(response.content)

        else:

            st.warning("Please enter a question.")