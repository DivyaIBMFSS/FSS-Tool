# app.py

import streamlit as st
from graphrag_core import process_question

st.set_page_config(page_title="GraphRAG Chatbot", layout="centered")
st.title("Welcome to your Chatbot")

st.markdown("Please input textual questions:")

question = st.text_input("🧠 Enter your question:")

if st.button("Submit"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Fetching your answer..."):
            answer = process_question(question)
        st.markdown("### 💬 Answer")
        st.markdown(answer)
