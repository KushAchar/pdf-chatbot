import os
from dotenv import load_dotenv

load_dotenv()


from utils.loader import load_pdf
from utils.splitter import split_documents
documents = load_pdf("data/sample.pdf")
chunks = split_documents(documents)
print("Length of documents",len(documents))
print("Length of chunks",len(chunks))
