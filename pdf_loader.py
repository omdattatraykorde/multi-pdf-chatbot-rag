from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("data/sample.pdf")

documents = loader.load()

print("Total Pages:", len(documents))
print("\nFirst Page:\n")
print(documents[0].page_content[:1000])