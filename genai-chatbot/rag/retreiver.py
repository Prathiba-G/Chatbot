from rag.database import load_or_create_index
from rag.embedder import encode_query


# 🔍 Retrieve top-k documents from vector index
def get_relevant_documents(query: str, k: int = 5) -> list[str]:
    # Load index and document store
    index, doc_store = load_or_create_index()

    # Encode query into vector
    query_vec = encode_query(query)

    # Search index
    distances, indices = index.search(query_vec, k)

    # Return matched documents
    return [doc_store[i] for i in indices[0] if i < len(doc_store)]
