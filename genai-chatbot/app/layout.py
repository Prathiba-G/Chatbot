import streamlit as st

def show_header():
    st.markdown("<h1 style='text-align: center; color: #0066cc;'>🧠 GenAI Chatbot with RAG</h1>", unsafe_allow_html=True)
    st.markdown("Welcome to your intelligent assistant powered by Llama 3 and RAG.")
    st.divider()


def show_input_form():
    with st.form(key="query_form"):
        user_query = st.text_input("🔍 Enter your question:")
        submit_btn = st.form_submit_button("Ask")
    return user_query if submit_btn else None


def show_response_panel(response, context, prompt, metrics):
    st.subheader("📨 Response")
    st.write(response)

    with st.expander("📄 Retrieved Context"):
        for i, doc in enumerate(context, 1):
            st.markdown(f"**Doc {i}:** {doc}")

    with st.expander("🧩 Prompt Used"):
        st.code(prompt, language="markdown")

    with st.expander("📊 Evaluation Metrics"):
        for key, value in metrics.items():
            st.write(f"**{key}:** {value}")
