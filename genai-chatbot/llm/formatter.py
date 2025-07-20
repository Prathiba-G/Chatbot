import re


# 🧹 Clean and normalize LLM output
def clean_response(text: str) -> str:
    if not text:
        return "⚠️ No response generated."
    # Remove excessive whitespace and line breaks
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# 🎨 Format response for display
def format_for_ui(text: str) -> str:
    text = clean_response(text)
    # Optional: Add markdown styling or emoji cues
    if text.lower().startswith("answer:"):
        text = f"🧠 **{text}**"
    return text


# 🧪 Example usage
if __name__ == "__main__":
    raw = "Answer:\n\nThis is a sample response.\n\n\nIt has extra spacing."
    print(format_for_ui(raw))
