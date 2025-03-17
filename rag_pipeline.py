from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY in .env file")

# Load tokenizer and model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)

class DocumentStore:
    def __init__(self, embedding_model):
        self.documents = []
        self.embedding_model = embedding_model
        self.tokenized_docs = []
        self.embeddings = []
        self.bm25 = None
        self.index = None
        
    def add_documents(self, new_docs):
        # Add to document store
        doc_ids = list(range(len(self.documents), len(self.documents) + len(new_docs)))
        self.documents.extend(new_docs)
        
        # Update tokenized docs and BM25
        new_tokenized = [doc.split() for doc in new_docs]
        self.tokenized_docs.extend(new_tokenized)
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        # Update embeddings and FAISS
        new_embeddings = self.embedding_model.encode(new_docs, convert_to_numpy=True)
        self.embeddings = np.vstack([self.embeddings, new_embeddings]) if len(self.embeddings) > 0 else new_embeddings
        
        # Rebuild or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(new_embeddings.shape[1])
        self.index.add(new_embeddings)
        
        return doc_ids
    
    def retrieve_bm25(self, query, top_n=2):
        if not self.bm25:
            return []
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [self.documents[i] for i in top_indices]
    
    def retrieve_faiss(self, query, top_n=2):
        if self.index is None or len(self.documents) == 0:
            return []
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(query_embedding, min(top_n, len(self.documents)))
        return [self.documents[i] for i in indices[0]]
    
    def hybrid_retrieval(self, query, top_n=3):
        bm25_results = self.retrieve_bm25(query, top_n)
        faiss_results = self.retrieve_faiss(query, top_n)
        
        # Combine and deduplicate results
        all_results = []
        seen_docs = set()
        
        for doc in bm25_results + faiss_results:
            if doc not in seen_docs:
                all_results.append(doc)
                seen_docs.add(doc)
        
        return all_results[:top_n]

# Initialize document store
doc_store = DocumentStore(embedding_model)

# Sample research documents
documents = [
    "Neural networks have revolutionized AI research.",
    "Tokenization is crucial for text processing in NLP.",
    "Adversarial robustness is a key challenge in AI models.",
    "FAISS enables efficient similarity search over embeddings."
]

# Add documents to the store
doc_store.add_documents(documents)

# Initialize LLM
try:
    llm = OpenAI(model_name="gpt-4", api_key=api_key)
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

def summarize_docs(docs):
    if not llm:
        return "LLM not initialized properly"
    
    try:
        return llm(f"Summarize these documents: {docs}")
    except Exception as e:
        return f"Error during summarization: {str(e)}"

def process_query(query, top_n=3):
    # 1. Retrieve relevant documents
    retrieved_docs = doc_store.hybrid_retrieval(query, top_n)
    
    # 2. Format prompt with retrieved context
    prompt = f"""
    Based on the following context, please answer the question:
    
    Context:
    {' '.join(retrieved_docs)}
    
    Question: {query}
    """
    
    # 3. Generate response with LLM
    try:
        response = llm(prompt)
        return {
            "answer": response,
            "retrieved_docs": retrieved_docs,
            "status": "success"
        }
    except Exception as e:
        return {
            "answer": None,
            "retrieved_docs": retrieved_docs,
            "status": "error",
            "error": str(e)
        }

def feedback_loop(query, feedback, retrieved_docs):
    """
    Incorporate feedback to improve retrieval quality
    
    Args:
        query: The original query
        feedback: "positive" or "negative"
        retrieved_docs: List of documents that were retrieved
    """
    if feedback == "positive":
        doc_store.add_documents(retrieved_docs)
        print(f"Added {len(retrieved_docs)} documents to the knowledge base")

# Example query and retrieval
query = "How does FAISS improve similarity search?"
result = process_query(query)

print("\n=== RAG Results ===")
print(f"Query: {query}")
print("\nRetrieved Documents:")
for i, doc in enumerate(result["retrieved_docs"]):
    print(f"{i+1}. {doc}")
print("\nAnswer:", result["answer"])