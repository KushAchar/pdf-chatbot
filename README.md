# pdf-chatbot

# A Retrieval‑Augmented Generation (RAG) pipeline turns a PDF into a searchable knowledge base by: splitting the PDF into small overlapping chunks, converting each chunk into a vector embedding, indexing those vectors in FAISS for fast semantic search, and at query time retrieving only the most relevant chunks to send to an LLM (Gemini) so answers are grounded in the document. 

Overview
Goal: Answer “What is deadlock?” using only the PDF content.

High‑level flow: PDF → chunking → embeddings → FAISS index → query embedding → retrieve top‑k chunks → send those chunks + question to Gemini → Gemini answers. 

Step‑by‑step pipeline (detailed)
Load PDF

Use a PDF loader (PyPDFLoader, PyMuPDF) to extract page text and metadata (page number, filename). 

Chunk the text

Split into small, semantically coherent chunks (typical: 500–800 characters or ~200–800 tokens) with overlap (e.g., 100–150 tokens) so context crossing boundaries is preserved. Overlap reduces boundary loss for concepts like “deadlock.” 

Convert chunks to embeddings

Use a sentence/embedding model (OpenAI embeddings, Sentence‑Transformers like all‑MiniLM) to map each chunk to a dense vector (e.g., 384–1536 dims). Each chunk → one vector stored alongside metadata. 

Index with FAISS

Build a FAISS index (IndexFlatL2, HNSW, or IVF+PQ for large corpora) for fast nearest‑neighbor search. Persist the index to disk so you don’t recompute embeddings every run. 

Query time: semantic retrieval

Embed the user question, run FAISS search (e.g., k = 3–10) to return the most relevant chunks. Only these chunks are forwarded to Gemini as context. This keeps token usage and cost low and reduces hallucination. 

Prompt Gemini with retrieved context

Construct a prompt that explicitly instructs Gemini to answer using only the provided chunks and to cite chunk metadata (page numbers). Example: “Use only the following excerpts to answer. If not found, say ‘not in document.’” 

Implementation details and best practices
Chunk size & overlap: 800 chars / 150 overlap is common; tune for your PDF’s density. 

Embedding model choice: trade cost vs. quality; miniLM is cheap and fast, OpenAI models give higher quality. 

FAISS index type: use IndexFlatL2 for small datasets; HNSW/IVF+PQ for millions of chunks. 

k selection: start with k=5, show retrieved chunks to user for transparency. 

Operational considerations
Latency: host FAISS and embedding service close to users (for you in Karkal, host in an India region) to reduce round‑trip time.

Privacy: keep sensitive PDFs and indices on private infrastructure; encrypt persisted FAISS files.

Updates: when PDF changes, re‑chunk and re‑embed only changed pages to save cost.

Hallucination mitigation: force Gemini to quote chunk IDs/pages and return “not found” when unsupported.