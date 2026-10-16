from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyMuPDFLoader("data/sample.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("=" * 50)
print("Total Chunks:", len(chunks))
print("=" * 50)

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}")
    print("-" * 50)
    print(chunk.page_content)
    