import os
from groq import Groq
from llm.formatter import format_for_ui

# 🔐 Load API key from environment or secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 🚀 Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# 🧠 Model name (you can change this based on availability)
MODEL_NAME = "llama3-8b-8192"


# 💬 Send prompt to Groq API and get response
def get_llm_response(prompt: str) -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
        )
        raw_response = chat_completion.choices[0].message.content
        return format_for_ui(raw_response)

    except Exception as e:
        return f"❌ Groq API error: {str(e)}"
