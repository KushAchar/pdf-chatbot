import os
from dotenv import load_dotenv

load_dotenv()


from utils.loader import load_pdf

documents = load_pdf("data/sample.pdf")
print("length of documents:", len(documents))
print("first document:", documents[0].page_content)