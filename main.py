import streamlit as st
import cohere
import os
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tempfile

# Set your Cohere API key here
COHERE_API_KEY = "nocojkkDxtVHUgFlrKwwh9fPTBwIsBWOxfx9T7Yz"
co = cohere.Client(COHERE_API_KEY)

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Customize page
st.set_page_config(
    page_title="Myodetox Assistant",
    page_icon="🧠",
    layout="wide"
)

# Sidebar with branding
with st.sidebar:
    st.image("Product Evolve.png", use_column_width=True)
    st.markdown("**Powered by Product Evolve & Cohere**")
    st.markdown("📧 contact@productevolve.com\n\n🌐 www.productevolve.com")

st.title("🧠 Myodetox Assistant (Text + PDF Search)")
st.markdown("Built by **Product Evolve** • Powered by **Cohere**")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []
    st.session_state.pdf_embeddings = []

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    chunks = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            chunks.extend([chunk.strip() for chunk in text.split("\n") if chunk.strip()])

    st.session_state.pdf_chunks = chunks
    st.session_state.pdf_embeddings = embedder.encode(chunks)

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_question = st.text_input("Type your question below:", key="input_field")
    submit = st.form_submit_button("Ask")

# Handle Q&A
if submit and user_question:
    st.session_state.chat_history.append(("You", user_question))

    # If PDF uploaded, use embeddings to find relevant context
    if st.session_state.pdf_chunks:
        question_embedding = embedder.encode([user_question])
        similarities = cosine_similarity(question_embedding, st.session_state.pdf_embeddings)[0]
        top_idx = similarities.argmax()
        context = st.session_state.pdf_chunks[top_idx]
        prompt = f"Context: {context}\n\nQuestion: {user_question}\n\nAnswer:"
    else:
        prompt = f"Answer the following question clearly:\n\n{user_question}"

    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=300,
        temperature=0.3
    ).generations[0].text.strip()

    st.session_state.chat_history.append(("Bot", response))

# Show chat
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"🧑 **{sender}:** {message}", unsafe_allow_html=True)
    else:
        st.markdown(f"🤖 **{sender}:** {message}", unsafe_allow_html=True)

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

