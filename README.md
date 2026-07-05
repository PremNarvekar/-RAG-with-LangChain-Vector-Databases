# RAG with LangChain & Vector Databases

A hands-on repository documenting my implementation and experiments while learning Retrieval-Augmented Generation (RAG) using LangChain, ChromaDB, Hybrid Search, LangGraph, FastAPI, and production-grade RAG architectures.

This repository follows the **Production RAG with LangChain & Vector Databases** course and includes practical implementations of modern RAG techniques, optimization strategies, security, observability, and deployment workflows.

---

## Tech Stack

- Python
- LangChain
- ChromaDB
- Google Gemini Embeddings
- BM25
- Hybrid Search
- LangSmith
- LangGraph
- FastAPI
- Supabase
- PGVector
- Python Dotenv

---

## Repository Structure

```
.
├── document_loaders.py
├── hybrid/
├── main.py
├── .env
├── pyproject.toml
└── README.md
```

---

## Topics Covered

### Foundations

- Introduction to Retrieval-Augmented Generation (RAG)
- Complete RAG Architecture
- Development Environment Setup
- Document Loading
- Document Processing Pipeline
- Embeddings Deep Dive
- Vector Database Fundamentals

### Vector Search

- Chroma Vector Database
- Similarity Search
- Search Scoring
- Metadata Filtering
- Building a Basic RAG Pipeline

### Retrieval Techniques

- Dense Retrieval
- Sparse Retrieval (BM25)
- Hybrid Search
- Reciprocal Rank Fusion (RRF)
- Token Budgeting

### Debugging & Optimization

- RAG Debugging
- Retrieval Optimization
- Context Optimization
- Cost Analysis
- Scaling Strategies

### Production Engineering

- LangSmith Observability
- Production Monitoring
- FastAPI Integration
- Production Deployment
- Security Layer
- Security Checklist

### Advanced RAG

- Long Context Models vs RAG
- Contextual Retrieval
- Late Chunking vs Early Chunking
- Agentic RAG
- Self-Correcting Retrieval
- GraphRAG
- Multi-hop Reasoning
- Multimodal RAG (ColPali)
- Future of RAG Systems

---

## Implementations

This repository contains implementations of:

- Document Loaders
- Text Chunking
- Embedding Generation
- Chroma Vector Store
- Similarity Search
- Basic RAG
- Hybrid Search (Vector + BM25)
- Reciprocal Rank Fusion
- LangSmith Integration
- FastAPI Backend
- LangGraph Agent
- Production Security Layer

More implementations will be added as I progress through the course.

---

## Learning Goals

The purpose of this repository is to understand how production RAG systems are built, optimized, monitored, and deployed.

Key focus areas include:

- Retrieval quality
- Vector databases
- Search optimization
- Production engineering
- AI observability
- Security
- Agentic workflows
- Modern RAG architectures

---

## Course Progress

- [x] RAG Fundamentals
- [x] Document Loading
- [x] Embeddings
- [x] ChromaDB
- [x] Similarity Search
- [x] Basic RAG
- [x] Hybrid Search
- [ ] Token Budgeting
- [ ] LangSmith
- [ ] RAG Optimization
- [ ] Scaling RAG
- [ ] Production Hosting
- [ ] FastAPI
- [ ] Security Layer
- [ ] LangGraph
- [ ] Agentic RAG
- [ ] GraphRAG
- [ ] Multimodal RAG

---

## Resources

- LangChain
- ChromaDB
- Google Gemini API
- LangSmith
- LangGraph
- FastAPI
- Supabase
- PGVector

---

## License

This repository is intended for educational purposes and personal learning.