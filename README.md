
# Multi-PDF Chatbot using Retrieval-Augmented Generation (RAG)

## About the Project

The Multi-PDF Chatbot is a Python-based application that allows users to interact with multiple PDF documents through natural language questions. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from documents and generate context-aware answers using a language model.

The project explores document processing, text embeddings, semantic search, and Large Language Model (LLM) integration.

## Features

- Upload and process PDF documents
- Extract text and split it into smaller chunks
- Generate embeddings for document text
- Retrieve relevant information using vector search
- Ask questions about document content
- Generate document summaries and suggested questions
- Interactive web interface built with Streamlit
- Chat history and research evaluation utilities

## Technology Stack

- **Programming Language:** Python
- **Frontend:** Streamlit
- **RAG Framework:** LangChain
- **Embeddings:** Hugging Face Sentence Transformers
- **Vector Search:** FAISS
- **Language Model Integration:** Google Gemini and Groq
- **Document Processing:** PDF text extraction libraries

## How It Works

1. **Document Upload:** Users provide PDF documents.
2. **Text Extraction:** Text is extracted from the uploaded files.
3. **Text Chunking:** The extracted content is divided into smaller chunks.
4. **Embedding Generation:** Text chunks are converted into vector embeddings.
5. **Information Retrieval:** Relevant chunks are retrieved based on the user's question.
6. **Response Generation:** A language model generates an answer using the retrieved context.

## Project Structure

```text
Multi_PDF_Chatbot/
├── app.py
├── components/
├── chat_history.py
├── experiment_manager.py
├── ingest.py
├── llm_factory.py
├── pdf_loader.py
├── rag.py
├── rag_chatbot.py
├── requirements.txt
├── research_evaluator.py
├── research_logger.py
├── retriever.py
├── suggestions.py
├── test_api.py
├── text_splitter.py
├── utils.py
├── vector_store.py
└── voice_input.py
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/omdattatraykorde/multi-pdf-chatbot-rag.git
cd multi-pdf-chatbot-rag
```

### 2. Create a Virtual Environment

For Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add the API keys required by your configured language model provider.

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder values with your own API keys. Never commit your `.env` file or publish API credentials.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Learning Outcomes

- Understanding Retrieval-Augmented Generation (RAG)
- Working with text embeddings and vector databases
- Implementing semantic retrieval
- Integrating language models into Python applications
- Building interactive applications with Streamlit
- Processing and retrieving information from PDF documents

## Future Improvements

- Improve retrieval accuracy and ranking
- Add source citations for generated answers
- Evaluate answer quality and retrieval performance
- Support additional document formats
- Deploy the application for online access

## Author

**Om Korde**

Data Science Engineering Student

- GitHub: [omdattatraykorde](https://github.com/omdattatraykorde)
- Project Repository: [Multi-PDF Chatbot using RAG](https://github.com/omdattatraykorde/multi-pdf-chatbot-rag)

---

This project was developed to explore document-based question answering using Retrieval-Augmented Generation.
