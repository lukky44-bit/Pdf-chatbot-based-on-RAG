# app.py
import streamlit as st
from chatbot import build_prompt, ask_llm, retrieve_docs

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="centered")

st.title("📄 PDF Chatbot (RAG)")
st.write("Ask questions from your uploaded PDFs")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
query = st.chat_input("Ask something from your PDFs...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # RAG pipeline
    with st.spinner("Searching PDFs and generating answer..."):
        docs = retrieve_docs(query, k=8)  # increase k for better coverage
        prompt = build_prompt(query, docs)
        answer = ask_llm(prompt)

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

        with st.expander("📚 Sources used"):
            for d in docs:
                st.write(f"- {d.metadata.get('source', 'Unknown')}")
