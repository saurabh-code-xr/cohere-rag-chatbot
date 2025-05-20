import os
from dotenv import load_dotenv
import cohere
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# Load knowledge base
with open("myodetox_embeddings.json", "r") as f:
    embedded_docs = json.load(f)

doc_vectors = [doc["embedding"] for doc in embedded_docs]

# Function to return RAG answer
def get_rag_answer(user_input):
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

    return response.generations[0].text.strip()


