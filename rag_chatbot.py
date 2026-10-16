import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load Gemini API Key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Load Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.3
)

# Load Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS Index
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("PDF Chatbot Ready!")
print("Type 'exit' to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not present in the context, say:
"I couldn't find that information in the uploaded PDF."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nBot:", response.content)
    print("-" * 60)