import pytest
from app.controller import handle_query


# 🧪 End-to-end test case
def test_full_rag_pipeline():
    query = "What is Retrieval-Augmented Generation?"
    result = handle_query(query)

    # ✅ Basic structure checks
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "response" in result, "Missing LLM response"
    assert "context" in result, "Missing retrieved context"
    assert "prompt" in result, "Missing constructed prompt"
    assert "metrics" in result, "Missing evaluation metrics"

    # ✅ Content checks
    assert len(result["context"]) > 0, "No documents retrieved"
    assert len(result["response"]) > 20, "Response too short"
    assert result["metrics"]["semantic_similarity"] > 0.5, "Low semantic similarity"
    assert result["metrics"]["response_quality"] > 0.5, "Low response quality"
