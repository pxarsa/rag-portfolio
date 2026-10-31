# Retrieval-Augmented Generation (RAG)

A simple RAG system built with Python, Sentence Transformers, and Ollama.

## Overview

This project retrieves relevant information from a local knowledge base using semantic search and uses a local Large Language Model (LLM) to generate answers based on the retrieved context.

## How It Works

1. **Chunking:** Splits the knowledge base into smaller text sections.
2. **Embeddings:** Converts text chunks into numerical vectors using Sentence Transformers.
3. **Similarity Search:** Measures semantic similarity between the user's question and the text chunks.
4. **Top-K Retrieval:** Retrieves the three most relevant chunks.
5. **Generation:** Uses Ollama to generate an answer based on the retrieved context.

## Technologies

- Python
- Sentence Transformers
- Ollama
- Llama 3.2 (3B)
- Cosine Similarity

## Requirements

- Python 3.10+
- Ollama installed and running
- Llama 3.2 model downloaded
- Required Python packages: `sentence-transformers`, `ollama`

## Run the Project

```bash
pip install sentence-transformers ollama
ollama pull llama3.2:3b
python rag_portfolio.py
```

Make sure `knowledge.txt` is in the same directory as `rag_portfolio.py`.

## Project Structure

```text
rag-portfolio/
├── rag_portfolio.py
├── knowledge.txt
└── README.md
```

## Purpose

This project demonstrates the fundamental concepts of Retrieval-Augmented Generation, semantic search, embeddings, and local LLM integration.