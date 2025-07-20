import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# 🧠 Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 📁 Paths
INDEX_PATH = "data/index/vector.index"
DOC_STORE_PATH = "data/index/doc_store.pkl"


# 🧱 Create or load FAISS index
def load_or_create_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(DOC_STORE_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOC_STORE_PATH, "rb") as f:
            doc_store = pickle.load(f)
        print("✅ Loaded existing index.")
    else:
        index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
        doc_store = []
        print("🆕 Created new index.")
    return index, doc_store


# 📦 Add documents to index
def add_documents(index, doc_store, documents):
    embeddings = embedder.encode(documents)
    index.add(embeddings)
    doc_store.extend(documents)
    save_index(index, doc_store)


# 💾 Save index and doc store
def save_index(index, doc_store):
    faiss.write_index(index, INDEX_PATH)
    with open(DOC_STORE_PATH, "wb") as f:
        pickle.dump(doc_store, f)
    print("💾 Index saved.")


# 🔍 Retrieve top-k documents
def retrieve_documents(index, doc_store, query, k=5):
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, k)
    return [doc_store[i] for i in indices[0] if i < len(doc_store)]
