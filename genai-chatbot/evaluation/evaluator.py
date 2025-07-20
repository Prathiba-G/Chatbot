import os
import json
from datetime import datetime


# 📁 Path to store feedback logs
FEEDBACK_LOG_PATH = "data/feedback_log.json"


# 📝 Save feedback to file
def save_feedback(user_query, llm_response, user_rating, user_comment=""):
    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": user_query,
        "response": llm_response,
        "rating": user_rating,
        "comment": user_comment
    }

    # Load existing feedback
    if os.path.exists(FEEDBACK_LOG_PATH):
        with open(FEEDBACK_LOG_PATH, "r") as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []

    # Append new entry
    feedback_data.append(feedback_entry)

    # Save back to file
    with open(FEEDBACK_LOG_PATH, "w") as f:
        json.dump(feedback_data, f, indent=2)

    print("✅ Feedback saved successfully.")


# 🧪 Example usage
if __name__ == "__main__":
    save_feedback(
        user_query="What is RAG in GenAI?",
        llm_response="RAG stands for Retrieval-Augmented Generation...",
        user_rating=4.5,
        user_comment="Very helpful explanation!"
    )
