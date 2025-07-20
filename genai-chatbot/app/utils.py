import os
import numpy as np
import pandas as pd
import logging
import time


# 💬 Timer utility
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = round(end - start, 2)
        print(f"⏱️ {func.__name__} executed in {duration} seconds")
        return result
    return wrapper


# 📁 Safe directory creator
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 📊 Pretty print dictionary
def pretty_print(d: dict, title: str = "Debug Info"):
    print(f"\n🔍 {title}")
    for key, val in d.items():
        print(f" - {key}: {val}")


# 🧹 Text cleaner
def clean_text(text: str) -> str:
    return text.strip().replace("\n", " ").replace("\t", " ")


# 📄 CSV loader
def load_csv(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()


# 📦 List flattener
def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]
