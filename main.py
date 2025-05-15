import streamlit as st
from streamlit_mic_recorder import mic_recorder
import cohere
import os
import json
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

# --- API KEY CONFIGURATION ---
COHERE_API_KEY = "nocojkkDxtVHUgFlrKwwh9fPTBwIsBWOxfx9T7Yz"
co = cohere.Client(COHERE_API_KEY)

# --- EMBEDDING MODEL CONFIGURATION ---
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# --- SIDEBAR ---
with st.sidebar:
    st.image("Product Evolve.png", width=180)
    st.markdown("### Powered by Product Evolve & Cohere")
    st.markdown("📩 Contact: hello@productevolve.com")

# --- MAIN HEADER ---
st.title("🧠 Cohere RAG Chatbot")
st.caption("Upload PDFs and ask questions based on the content.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

# --- CONTEXT EXTRACTION FROM PDF ---
if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    sentences = text.split(". ")
    corpus_embeddings = embedder.encode(sentences, convert_to_tensor=True)

    st.success(f"✅ PDF '{uploaded_file.name}' uploaded and processed!")

    # --- CHAT SECTION ---
    st.markdown("### 💬 Ask a question based on your PDF")
    query = st.text_input("Ask something:", placeholder="e.g., What is the summary of section 2?")

    if query:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_idx = np.argmax(cos_scores.cpu().numpy())
        matched_sentence = sentences[top_idx]

        # Generate response with Cohere
        response = co.generate(
            model='command-r-plus',
            prompt=f"Answer the question based on this context:\n\nContext: {matched_sentence}\n\nQuestion: {query}",
            max_tokens=150
        )

        st.markdown("#### 🤖 Answer")
        st.write(response.generations[0].text.strip())

# --- OPTIONAL: MIC INPUT (IF NEEDED LATER) ---
with st.expander("🎙️ Voice Input (Optional)"):
    audio_bytes = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop", key="recorder")
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

# --- FOOTER ---
st.markdown("---")
st.markdown("© 2025 Product Evolve | Built with ❤️ using Cohere + Streamlit")
