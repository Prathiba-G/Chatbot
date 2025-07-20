import os
from arize.pandas.logger import Client
import pandas as pd
from datetime import datetime


# 🔐 Load credentials from environment or secrets
ARIZE_SPACE_KEY = os.getenv("ARIZE_SPACE_KEY")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")

# 🚀 Initialize Arize client
arize_client = Client(
    space_key=ARIZE_SPACE_KEY,
    api_key=ARIZE_API_KEY,
)

# 🧪 Log evaluation metrics
def log_evaluation(user_query, retrieved_docs, llm_response, latency_ms=0):
    timestamp = datetime.utcnow().isoformat()

    # Create a DataFrame with evaluation data
    eval_df = pd.DataFrame([{
        "timestamp": timestamp,
        "query": user_query,
        "retrieved_context": " | ".join(retrieved_docs),
        "llm_response": llm_response,
        "latency_ms": latency_ms,
        "relevance_score": compute_relevance(user_query, retrieved_docs),
        "response_quality": score_response(llm_response),
    }])

    # Log to Arize
    arize_client.log_evaluations_sync(
        dataframe=eval_df,
        project_name="genai-chatbot"
    )


# 🧠 Dummy scoring functions (customize as needed)
def compute_relevance(query, docs):
    return min(1.0, len(docs) / 5)  # Example: scale relevance by # of docs

def score_response(response):
    return 1.0 if len(response) > 50 else 0.5  # Example: basic length-based score
