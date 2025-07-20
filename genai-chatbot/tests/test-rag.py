import pytest
from rag.rag_chain import run_rag_chain


# 🧪 Test basic RAG response
def test_rag_basic_query():
    query = "What is Retrieval-Augmented Generation?"
    result = run_rag_chain(query)

    assert isinstance(result, dict), "Result should be a dictionary"
    assert "response" in result, "Missing LLM response"
    assert "context" in result, "Missing retrieved context"
    assert "prompt" in result, "Missing constructed prompt"

    assert len(result["context"]) > 0, "No documents retrieved"
    assert len(result["response"]) > 20, "Response too short"


# 🧪 Test with domain-specific query
def test_rag_vector_db_query():
    query = "How does FAISS perform vector search?"
    result = run_rag_chain(query)

    assert "FAISS" in result["response"], "Response should mention FAISS"
    assert result["metrics"]["semantic_similarity"] > 0.5, "Low semantic similarity"
    assert result["metrics"]["response_quality"] > 0.5, "Low response quality"


# 🧪 Test fallback behavior
def test_rag_empty_query():
    query = "   "
    result = run_rag_chain(query)

    assert isinstance(result, dict) or isinstance(result, str), "Unexpected result type"
    if isinstance(result, str):
        assert "Please enter a valid query" in result
