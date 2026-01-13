import requests
from vectorstore import load_vector_store

MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"


def retrieve_docs(query, k=3):
    vectordb = load_vector_store()
    results = vectordb.similarity_search(query, k=k)

    return results


def build_prompt(query, docs):
    context = "\n\n".join(
        [f"Source ({i + 1}): {doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
You are a PDF assistant. Use only the information from the context to answer.
If the answer is not in the context, say "I don't know".
Also dont give what u r thinking steps just give the final answer.
Context:
{context}

Question:
{query}

Answer:
"""
    return prompt


def ask_llm(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}"


# if __name__ == "__main__":
#     print("Manual RAG PDF Chatbot Ready. Type 'exit' to quit.\n")

#     while True:
#         query = input("You: ")
#         if query.lower() == "exit":
#             break

#         docs = retrieve_docs(query, k=8)
#         prompt = build_prompt(query, docs)
#         answer = ask_llm(prompt)

#         print("\nBot:", answer.strip())

#         print("\nSources:")
#         for d in docs:
#             print("-", d.metadata.get("source", "Unknown"))
#         print("-" * 60)
