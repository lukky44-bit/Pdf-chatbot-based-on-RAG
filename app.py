# app.py

import streamlit as st
import os
from chatbot2 import get_qa_chain
from vectorstore import rebuild_vector_store

# ------------------ CONFIG ------------------
PDF_FOLDER = "data"
os.makedirs(PDF_FOLDER, exist_ok=True)

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="centered")
st.title("📄 PDF Chatbot (RAG)")

# ------------------ SIDEBAR (UPLOAD AREA) ------------------
st.sidebar.header("📂 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
)

if st.sidebar.button("Process PDFs"):
    if uploaded_files:
        with st.spinner("Saving PDFs and rebuilding vector database..."):
            # Save uploaded PDFs
            for file in uploaded_files:
                file_path = os.path.join(PDF_FOLDER, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())

            # Rebuild vector database safely
            rebuild_vector_store(PDF_FOLDER)

        st.sidebar.success("✅ PDFs uploaded and Vector DB rebuilt successfully!")
    else:
        st.sidebar.warning("⚠️ Please upload at least one PDF before processing.")

# ------------------ CHAT HISTORY ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ CHAT INPUT ------------------
query = st.chat_input("Ask something from your PDFs...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run RAG pipeline
    with st.spinner("🔍 Searching PDFs and generating answer..."):
        try:
            qa_chain = get_qa_chain()
            result = qa_chain.invoke(query)
            answer = result["result"]
            docs = result.get("source_documents", [])
        except Exception as e:
            answer = f"❌ Error occurred: {e}"
            docs = []

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

        # Show sources
        if docs:
            with st.expander("📚 Sources used"):
                for d in docs:
                    st.write(f"- {d.metadata.get('source', 'Unknown')}")
