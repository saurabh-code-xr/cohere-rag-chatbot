import streamlit as st
import cohere
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder

# Load Cohere API key
co = cohere.Client("nocojkkDxtVHUgFlrKwwh9fPTBwIsBWOxfx9T7Yz")  # Replace with your actual key

# Load embedded knowledge base
with open("myodetox_embeddings.json", "r") as f:
    embedded_docs = json.load(f)
doc_vectors = [doc["embedding"] for doc in embedded_docs]

# --- UI Configuration ---
st.set_page_config(page_title="Product Evolve - Myodetox Voice Assistant", page_icon="🌀")

st.markdown(
    """
    <style>
    .title { font-size: 32px; font-weight: 700; }
    .subtitle { font-size: 18px; color: gray; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True
)
st.markdown("<div class='title'>🎙️ Myodetox Voice Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Built by <b>Product Evolve</b> • Powered by Cohere</div>", unsafe_allow_html=True)

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Input: Text or Voice ---
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Your question:", key="text_input")
with col2:
    audio = mic_recorder(start_prompt="🎤 Speak", stop_prompt="⏹️ Stop", just_once=True, use_container_width=True)
    if audio:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio["filename"]) as source:
            audio_data = recognizer.record(source)
            try:
                user_input = recognizer.recognize_google(audio_data)
                st.success(f"You said: {user_input}")
            except sr.UnknownValueError:
                st.warning("Sorry, we couldn't understand the audio.")

# --- Process the input ---
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

# --- Display conversation ---
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")

# --- Clear Chat Button ---
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# --- Footer Branding ---
st.markdown(
    """
    <hr style='margin-top: 50px;'>
    <div style='text-align: center; font-size: 14px; color: gray;'>
        © 2025 Product Evolve • All rights reserved.<br>
        Built with ❤️ using Streamlit and Cohere
    </div>
    """, unsafe_allow_html=True
)
