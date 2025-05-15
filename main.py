import streamlit as st
import cohere
import os
import base64
from PyPDF2 import PdfReader

# Load API key
COHERE_API_KEY = "nocojkkDxtVHUgFlrKwwh9fPTBwIsBWOxfx9T7Yz"
co = cohere.Client(COHERE_API_KEY)

# App Configuration
st.set_page_config(page_title="Evolve Concierge", layout="wide")

# Sidebar Branding
with st.sidebar:
    st.image("Product Evolve.png", use_column_width=True)
    st.markdown("### Evolve Concierge")
    st.markdown("**Ask anything. Upload docs. Get answers.**")
    theme = st.radio("Choose Mode:", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("<style>body { background-color: #0e1117; color: white; }</style>", unsafe_allow_html=True)
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []

# Transcript download
def get_text_download_link(text, filename="transcript.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Transcript</a>'
    return href

# Memory and session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Text Extraction
def extract_text_from_pdf(uploaded_files):
    full_text = ""
    for file in uploaded_files:
        reader = PdfReader(file)
        for page in reader.pages:
            full_text += page.extract_text() or ""
    return full_text

# Title
st.title("🤖 Evolve Concierge")

# File Upload and Preview
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
pdf_text = ""
if uploaded_files:
    st.subheader("📄 Preview of Uploaded PDF(s):")
    for file in uploaded_files:
        st.markdown(f"**{file.name}**")
    pdf_text = extract_text_from_pdf(uploaded_files)

# Question Input
question = st.text_input("💬 Ask your question")

# Q&A Handling
if question:
    context = pdf_text or ""
    history = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
    prompt = f"{history}\n\nContext:\n{context}\n\nQ: {question}\nA:"
    
    with st.spinner("Thinking..."):
        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=400,
            temperature=0.3
        )
        answer = response.generations[0].text.strip()
    
    st.markdown("### 🤖 Answer:")
    st.write(answer)
    st.code(answer, language="text")
    st.button("📋 Copy to Clipboard", on_click=lambda: st.toast("Copied!", icon="✅"))

    # Save Q&A in session
    st.session_state.chat_history.append((question, answer))

# Display full chat
if st.session_state.chat_history:
    st.subheader("📝 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}: {q}**")
        st.markdown(f"*A{i+1}: {a}*")

    # Transcript Download Link
    transcript = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
    st.markdown(get_text_download_link(transcript), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("⚡ Powered by **Product Evolve** & **Cohere**")

