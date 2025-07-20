import pytest
from prompts.user_templates import build_prompt


# 🧪 Basic prompt construction test
def test_prompt_with_context():
    query = "What is Retrieval-Augmented Generation?"
    context = [
        "RAG combines retrieval with generation.",
        "It improves factual accuracy by grounding responses in documents."
    ]
    prompt = build_prompt(query, context)

    assert "Retrieval-Augmented Generation" in prompt
    assert "Context:" in prompt
    assert "- RAG combines retrieval with generation." in prompt
    assert "User:" in prompt
    assert "Assistant:" in prompt


# 🧪 Test with persona injection
def test_prompt_with_persona():
    query = "Explain Llama 3 in simple terms."
    context = ["Llama 3 is a large language model developed for natural language tasks."]
    persona = "GenAI Tutor"
    prompt = build_prompt(query, context, persona)

    assert "<persona type=\"GenAI Tutor\">" in prompt
    assert "Llama 3" in prompt
    assert "Assistant:" in prompt


# 🧪 Test with empty context
def test_prompt_empty_context():
    query = "What is FAISS?"
    context = []
    prompt = build_prompt(query, context)

    assert "Context:" in prompt
    assert "User: What is FAISS?" in prompt
    assert "Assistant:" in prompt
