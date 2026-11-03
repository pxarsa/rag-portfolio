# Retrieval-Augmented Generation (RAG)

A local Retrieval-Augmented Generation (RAG) system built with Python, Sentence Transformers, FAISS, and Ollama.

## Overview

This project implements a RAG pipeline that retrieves relevant information from a local knowledge base using semantic search and generates context-aware answers with a locally running Large Language Model (LLM).

The system uses vector embeddings and FAISS similarity search to find relevant text chunks before passing the retrieved context to the LLM.

## Architecture

The RAG pipeline consists of the following steps:

1. **Knowledge Base Loading:** Reads information from a local `knowledge.txt` file.
2. **Chunking:** Splits the knowledge base into smaller text sections.
3. **Embedding Generation:** Converts text chunks into 384-dimensional vectors using Sentence Transformers.
4. **Vector Indexing:** Stores embeddings in a FAISS index for efficient similarity search.
5. **Semantic Retrieval:** Converts the user's question into an embedding and retrieves the three most relevant chunks.
6. **Similarity Threshold:** Checks whether the retrieved information is sufficiently relevant before generating an answer.
7. **Answer Generation:** Sends the retrieved context to Ollama, which generates an answer using Llama 3.2 (3B).

## Key Features

- Semantic search using sentence embeddings.
- FAISS-based vector similarity search.
- Top-3 relevant chunk retrieval.
- Similarity threshold to reduce irrelevant answers.
- Persistent FAISS index storage.
- Local LLM integration using Ollama.
- Local knowledge base support.

## Technologies

- Python
- Sentence Transformers
- FAISS
- NumPy
- Ollama
- Llama 3.2 (3B)
- Cosine Similarity

## Requirements

- Python 3.10+
- Ollama installed and running
- Llama 3.2 (3B) model downloaded
- Required Python packages:
  - `sentence-transformers`
  - `faiss-cpu`
  - `numpy`
  - `ollama`

## Installation and Usage

Install the required Python packages:

```bash
pip install sentence-transformers faiss-cpu numpy ollama
```

Download the Llama model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running, then start the application:

```bash
python rag_portfolio.py
```

Place `knowledge.txt` in the same directory as `rag_portfolio.py`.

## Project Structure

```text
rag-portfolio/
├── rag_portfolio.py
├── knowledge.txt
├── chunks.json
├── README.md
└── .gitignore
```

The FAISS index is stored locally as `rag_index.faiss` and is excluded from Git tracking.

## Learning Objectives

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Text chunking
- Embeddings and vector representations
- Semantic similarity search
- Vector databases and FAISS indexing
- Retrieval quality control
- Local LLM integration

## Future Improvements

- Evaluate retrieval quality across a larger set of test questions.
- Improve chunking strategies.
- Add metadata filtering.
- Build a simple user interface.
