import re
import nltk
from typing import List

# 📦 Download NLTK sentence tokenizer if not already present
nltk.download("punkt", quiet=True)


# ✂️ Fixed-size chunking (by characters)
def chunk_by_chars(text: str, chunk_size: int = 500) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# 🧩 Sentence-based chunking
def chunk_by_sentences(text: str, max_tokens: int = 100) -> List[str]:
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    token_count = 0

    for sentence in sentences:
        tokens = sentence.split()
        if token_count + len(tokens) <= max_tokens:
            current_chunk += " " + sentence
            token_count += len(tokens)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            token_count = len(tokens)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# 🧠 Semantic chunking (optional: requires embeddings)
def chunk_semantically(text: str, embedder, threshold: float = 0.5) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    embeddings = embedder.encode(sentences)
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = embedder.similarity(embeddings[i - 1], embeddings[i])
        if sim > threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
