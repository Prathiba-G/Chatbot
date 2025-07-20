from rag.rag_chain import generate_rag_response
from prompts.user_templates import build_prompt
from llm.groq_api import get_llm_response
from evaluation.evaluator import evaluate_response
from streamlit.runtime.scriptrunner import get_script_run_ctx


def handle_query(user_query):
    if not user_query or not user_query.strip():
        return "⚠️ Please enter a valid query."

    try:
        # Step 1: Run RAG pipeline to retrieve relevant context
        context_docs = generate_rag_response(user_query)

        # Step 2: Build dynamic prompt using retrieved docs
        prompt = build_prompt(user_query, context_docs)

        # Step 3: Query LLM (Llama 3 via Groq)
        llm_answer = get_llm_response(prompt)

        # Step 4: Evaluate response with Arize or fallback
        metrics = evaluate_response(user_query, context_docs, llm_answer)

        return {
            "prompt": prompt,
            "context": context_docs,
            "response": llm_answer,
            "metrics": metrics
        }

    except Exception as e:
        return f"❌ Error processing query: {str(e)}"
