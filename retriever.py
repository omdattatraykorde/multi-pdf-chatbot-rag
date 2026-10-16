from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_database(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    db = load_database()
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
    while True:
        query = input("Ask a question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        docs = retriever.invoke(query)
        print("\nTop 3 Relevant Chunks\n")
        for i, doc in enumerate(docs, start=1):
            print("=" * 70)
            print(f"Chunk {i}")
            print("=" * 70)
            print(doc.page_content)
            print()
