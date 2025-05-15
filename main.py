import streamlit as st
from streamlit_mic_recorder import st_mic_recorder
import cohere
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime

# Load Cohere
co = cohere.Client("nocojkkDxtVHUgFlrKwwh9fPTBwIsBWOxfx9T7Yz")  # Replace securely in production

# Load Embeddings
with open("myodetox_embeddings.json", "r") as f:
    embedded_docs = json.load(f)

doc_vectors = [doc["embedding"] for doc in embedded_docs]

# -------------------- UI HEADER --------------------
st.set_page_config(page_title="Myodetox Voice Assistant", page_icon="🎙️")

st.markdown("""
    <h2 style='text-align: center;'>🎙️ Myodetox Voice Assistant</h2>
    <p style='text-align: center; font-size: 16px; color: gray;'>Built by <strong>Product Evolve</strong> • Powered by Cohere</p>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- INPUT --------------------
user_input = st.text_area("🧑‍💬 Type or speak your question below:", height=100, placeholder="e.g., What’s your cancellation policy?")

mic_audio = st_mic_recorder(start_prompt="🎤 Speak", stop_prompt="⏹️ Stop", just_once=True, key="mic")

if mic_audio and not user_input:
    user_input = mic_audio

# -------------------- RESPONSE --------------------
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
If the answer isn't in the context, say: "I'm not sure based on the available info."

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
    ).generations[0].text.strip()

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# -------------------- DISPLAY --------------------
st.markdown("<br>", unsafe_allow_html=True)

for sender, message in st.session_state.chat_history:
    bubble_color = "#f0f2f6" if sender == "You" else "#e6f7ec"
    emoji = "🧑‍💬" if sender == "You" else "🤖"
    st.markdown(f"""
    <div style='background-color:{bubble_color}; padding:10px; border-radius:10px; margin-bottom:5px;'>
        {emoji} <strong>{sender}</strong>: {message}
    </div>
    """, unsafe_allow_html=True)

# -------------------- ACTIONS --------------------
st.button("🧹 Clear Chat", on_click=lambda: st.session_state.chat_history.clear())

# -------------------- FOOTER --------------------
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 13px; color: gray;'>
        © 2025 <strong>Product Evolve</strong> • Built with ❤️ using Streamlit and <a href='https://cohere.com' target='_blank'>Cohere</a> <br>
        <em>This assistant does not provide medical advice. Contact the clinic directly for personal inquiries.</em>
    </div>
""", unsafe_allow_html=True)
