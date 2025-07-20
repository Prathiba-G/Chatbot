import json
import os

# 📄 Load few-shot examples from examples.json
EXAMPLES_PATH = "prompts/examples.json"

def load_examples():
    if os.path.exists(EXAMPLES_PATH):
        with open(EXAMPLES_PATH, "r") as f:
            return json.load(f)
    return []


# 🧠 Build dynamic prompt using context and query
def build_prompt(user_query: str, retrieved_docs: list, persona: str = None) -> str:
    context_block = "\n".join([f"- {doc}" for doc in retrieved_docs])
    examples = load_examples()

    # Optional persona tag
    persona_tag = f"<persona type=\"{persona}\">" if persona else ""

    # Few-shot examples (optional)
    example_block = ""
    for ex in examples[:2]:  # Limit to 2 examples for brevity
        example_block += f"{ex['role'].capitalize()}: {ex['content']}\n"

    # Final prompt construction
    prompt = f"""
{persona_tag}
You are a helpful assistant powered by RAG and Llama 3.

Context:
{context_block}

{example_block}
User: {user_query}
Assistant:"""

