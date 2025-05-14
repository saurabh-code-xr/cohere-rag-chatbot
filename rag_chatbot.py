import streamlit as st
import cohere
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load Cohere API key
co = cohere.Client("your-api-key-here")  # Replace with your actual API key

# Load knowledge base
with open("myodetox_embeddings.json", "r") as f:
    embedded_docs = json.load(f)

doc_vectors = [doc["embedding"] for doc in embedded_docs]

# Streamlit UI
st.set_page_config(page_title="Myodetox AI Assistant", page_icon="💬")
st.title("💬 Myodetox Clinic Chatbot")
st.write("Ask me anything about the clinic services or policies:")

user_input = st.text_input("Your question:")

if user_input:
    query_embed = co.embed(
        texts=[user_input],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    similarities = cosine_similarity([query_embed], doc_vectors)[0]
    top_index = int(np.argmax(similarities))
    context = embedded_docs[top_index]["content"]

    prompt = f"""
    You are a helpful assistant for a physiotherapy clinic. Use only the context below to answer the user's question.
    If the answer isn't found in the context, say "I'm not sure based on the available info."

    Context:
    {context}

    Question:
    {user_input}

    Answer:"""

    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=200,
        temperature=0.5
    )

    st.markdown(f"**Bot:** {response.generations[0].text.strip()}")
