from rag.database import load_or_create_index, retrieve_documents
from rag.embedder import encode_query
from prompts.user_templates import build_prompt
from llm.groq_api import get_llm_response


# 🔗 Full RAG pipeline: query ➝ retrieve ➝ prompt ➝ generate
def generate_rag_response(user_query: str, persona: str = None) -> list[str]:
    # Step 1: Load vector index and document store
    index, doc_store = load_or_create_index()

    # Step 2: Retrieve top-k relevant documents
    retrieved_docs = retrieve_documents(index, doc_store, user_query, k=5)

    return retrieved_docs


# 🧠 Generate final response from LLM
def run_rag_chain(user_query: str, persona: str = None) -> dict:
    # Step 1: Retrieve context
    context_docs = generate_rag_response(user_query, persona)

    # Step 2: Build prompt using context and query
    prompt = build_prompt(user_query, context_docs, persona)

    # Step 3: Get LLM response
    llm_response = get_llm_response(prompt)

    return {
        "query": user_query,
        "context": context_docs,
        "prompt": prompt,
        "response": llm_response
    }
