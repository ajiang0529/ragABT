# Advanced RAG System
A robust Retrieval-Augmented Generation (RAG) system that combines multiple retrieval methods (BM25 and FAISS) with LLM-based generation.
## Overview
This system enhances LLM responses by retrieving relevant context from a document store before generating answers. It features:

Hybrid retrieval combining lexical search (BM25) and semantic search (FAISS)
Document management through a centralized DocumentStore class
Integration with OpenAI's GPT-4 for answer generation
User feedback loop for continuous improvement

Requirements

Python 3.8+
Dependencies listed in requirements.txt
OpenAI API key

Installation

Clone the repository
Install dependencies:
Copypip install -r requirements.txt

Create a .env file in the project root with your OpenAI API key:
CopyOPENAI_API_KEY=your_api_key_here


Core Components
DocumentStore
Manages document storage, indexing, and retrieval operations:

Maintains document collection
Creates and updates BM25 index for lexical search
Manages FAISS index for vector similarity search
Provides hybrid retrieval combining both methods

Retrieval Methods

BM25 Retrieval: Token-based lexical search
FAISS Retrieval: Dense vector similarity search
Hybrid Retrieval: Combines and deduplicates results from both methods

Query Processing
The process_query function implements a complete RAG pipeline:

Retrieves relevant documents using hybrid retrieval
Constructs a prompt with the retrieved context
Sends the prompt to the LLM for answer generation
Returns a structured response with answer and retrieved documents

Feedback Loop
The system can improve over time by incorporating user feedback:

Positive feedback adds documents to the knowledge base
Both retrieval indices are updated accordingly

Usage Example
pythonCopy# Initialize with documents
documents = [
    "Neural networks have revolutionized AI research.",
    "Tokenization is crucial for text processing in NLP.",
    "Adversarial robustness is a key challenge in AI models.",
    "FAISS enables efficient similarity search over embeddings."
]
doc_store.add_documents(documents)

# Process a query
query = "How does FAISS improve similarity search?"
result = process_query(query)

# Provide feedback
feedback_loop(query, "positive", result["retrieved_docs"])
Extensions and Improvements
Potential enhancements for the system:

Document chunking for handling longer texts
Reranking step to improve retrieval precision
Metadata filtering for targeted retrieval
Persistent storage for document embeddings and indices
Web UI for interactive query answering
Multi-modal document support

License
MIT License
