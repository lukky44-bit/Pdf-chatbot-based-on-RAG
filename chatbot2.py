from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from vectorstore import load_vector_store
import os
from dotenv import load_dotenv

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()


def get_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)


def get_retriever(k=5):
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    return retriever


def get_template():
    template = """ you are a RAG assistant, you must only use the given context to answer, 
    if the user asks any genreal stuffs like good morning etc reply them with a correct reply
    
    Context:{context}

    Question:{question}
    """

    return PromptTemplate(template=template, input_variables=["context", "question"])


def get_qa_chain():
    llm = get_llm()
    retriever = get_retriever()
    prompt = get_template()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain


# if __name__ == "__main__":
#     qa = get_qa_chain()
#     print("RAG Chatbot using Groq + Chain is ready. Type 'exit' to quit.\n")

#     while True:
#         query = input("You: ")
#         if query.lower() == "exit":
#             break

#         result = qa.invoke(query)

#         print("\nBot:", result["result"])
