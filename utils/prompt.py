from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant.

Answer ONLY using the context below.

If the answer is not in the context,
reply:
"I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)