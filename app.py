import streamlit as st
import os
from chatbot2 import get_qa_chain
from vectorstore import rebuild_vector_store

# ------------------ CONFIG ------------------
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="Multimodal RAG Chatbot", page_icon="🤖", layout="centered"
)
st.title("🤖 Multimodal RAG Chatbot")
st.caption("Supports PDF, Images, TXT, and Scanned PDFs")

# ------------------ SIDEBAR (UPLOAD AREA) ------------------
st.sidebar.header("📂 Upload Files")

uploaded_files = st.sidebar.file_uploader(
    "Upload files (PDF, Images, TXT,Audio)",
    type=["pdf", "png", "jpg", "jpeg", "txt", ".mp3", ".wav", ".m4a", ".mp4"],
    accept_multiple_files=True,
)

if st.sidebar.button("Process Files"):
    if uploaded_files:
        with st.spinner("Saving files and rebuilding vector database..."):
            for file in uploaded_files:
                file_path = os.path.join(DATA_FOLDER, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())

            # This now processes:
            # Normal PDFs
            # Scanned PDFs → OCR
            # Images → OCR
            # TXT → direct read
            rebuild_vector_store(DATA_FOLDER)

            for file in os.listdir(DATA_FOLDER):
                file_path = os.path.join(DATA_FOLDER, file)
                if file == "Vector_store":
                    continue
                if os.path.isfile(file_path):
                    os.remove(file_path)

        st.sidebar.success("✅ Files uploaded and Vector DB rebuilt successfully!")
    else:
        st.sidebar.warning("⚠️ Please upload at least one file before processing.")

# ------------------ CHAT HISTORY ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ CHAT INPUT ------------------
query = st.chat_input("Ask something from your files...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("🔍 Searching knowledge base and generating answer..."):
        try:
            qa_chain = get_qa_chain()
            result = qa_chain.invoke(query)
            answer = result["result"]
            docs = result.get("source_documents", [])
        except Exception as e:
            answer = f"❌ Error occurred: {e}"
            docs = []

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

        if docs:
            with st.expander("📚 Sources used"):
                for d in docs:
                    st.write(f"- {d.metadata.get('source', 'Unknown')}")
