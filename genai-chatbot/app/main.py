import streamlit as st
from app.layout import show_header, show_input_form, show_response_panel
from app.controller import handle_query

# Set page config
st.set_page_config(page_title="GenAI Chatbot", layout="wide")

# UI Header
show_header()

# Input Form
user_query = show_input_form()

# Handle Query
if user_query:
    result = handle_query(user_query)

    # Show Results
    if isinstance(result, dict):
        show_response_panel(
            response=result["response"],
            context=result["context"],
            prompt=result["prompt"],
            metrics=result["metrics"]
        )
    else:
        st.error(result)
