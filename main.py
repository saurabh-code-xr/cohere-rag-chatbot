import streamlit as st
import cohere
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import tempfile

# ---- App Config ----
st.set_page_config(page_title="Myodetox AI Concierge", page_icon="🩺", layout="centered")

# ---- Styles ----
st.markdown("""
    <style>
    .user-bubble {
        background-color: #f0f0f5;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .bot-bubble {
        background-color: #e6f7ff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .footer {
        margin-top: 3rem;
        font-size: 0.8em;
        color: #888;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Header ----
st.title("🩺 Myodetox AI Concierge (Text + Voice)")
st.caption("Built by **Product Evolve** • Powered by **Cohere**")

# ---- Cohere API ----
co = cohere.Client("nocojkkDxtVHUgFlrKwwh9fPTBwIsBWOxfx9T7Yz")  # Replace with your API key

# ---- Load Embeddings ----
with open("myodetox_embeddings.json", "r") as f:
    embedded_docs = json.load(f)
doc_vectors = [doc["embedding"] for doc in embedded_docs]

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Input Section ----
col1, col2 = st.columns([4, 1])
with col1:
    user_text = st.text_input("Type or speak your question below:", key="text_input")
with col2:
    audio = mic_recorder(start_prompt="🎤 Speak", stop_prompt="⏹ Stop", key="mic")

user_input = ""

# ---- Process Audio ----
if audio and not user_text:
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(audio["bytes"])
        tmpfile.flush()
        with sr.AudioFile(tmpfile.name) as source:
            try:
                audio_data = recognizer.record(source)
                user_input = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                user_input = "Sorry, I couldn’t understand that."

if user_text:
    user_input = user_text

# ---- Query & Response ----
if user_input:
    # Step 1: Embed the query
    query_embed = co.embed(
        texts=[user_input],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    # Step 2: Similarity match
    similarities = cosine_similarity([query_embed], doc_vectors)[0]
    top_index = int(np.argmax(similarities))
    context = embedded_docs[top_index]["content"]

    # Step 3: Prompt + Generate
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

    # Step 4: Store in chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

    # Step 5: Clear input field
    st.experimental_rerun()

# ---- Display Chat ----
for sender, message in st.session_state.chat_history:
    css_class = "user-bubble" if sender == "You" else "bot-bubble"
    st.markdown(f'<div class="{css_class}"><b>{sender}:</b> {message}</div>', unsafe_allow_html=True)

# ---- Clear Chat Button ----
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# ---- Footer ----
st.markdown("""
<div class="footer">
    © 2025 Product Evolve • All rights reserved.<br>
    Built with ❤️ using Streamlit and Cohere
</div>
""", unsafe_allow_html=True)
