# PDF RAG Chatbot using LangChain + Ollama

A Retrieval-Augmented Generation (RAG) chatbot built with Python, LangChain, Ollama, and ChromaDB that lets users ask questions from a PDF document and get context-aware answers.

This project extracts text from a PDF, splits it into chunks, generates embeddings, stores them in a vector database, retrieves relevant content based on user queries, and uses an LLM to generate answers from the document context.

---

## Features

- Load and read PDF documents
- Extract text from PDF pages
- Split text into semantic chunks
- Generate embeddings using Ollama
- Store embeddings in Chroma vector database
- Perform similarity search over document content
- Ask natural language questions from the PDF
- Generate answers using Llama 3.2
- Streaming response output in terminal

---

## Tech Stack

- Python
- LangChain
- Ollama
- ChromaDB
- PyPDF
- RecursiveCharacterTextSplitter

---

## Project Workflow

```text
PDF
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
Chroma Vector DB
↓
Similarity Search
↓
LLM Response