import streamlit as st
import cohere
import os
import tempfile
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from gtts import gTTS
import live_voice_chat
from rag_chatbot import get_rag_answer

# --- Load API key from .env ---
load_dotenv()
COHERE_API_KEY = os.getenv("nocojkkDxtVHUgFlrKwwh9fPTBwIsBWOxfx9T7Yz")
co = cohere.Client(COHERE_API_KEY)

# --- Page Config ---
st.set_page_config(page_title="My Detox SPA - Agent", page_icon="💬")

# --- Sidebar ---
st.sidebar.image("Product Evolve.png", width=200)
st.sidebar.title("Evolve Concierge")
st.sidebar.markdown("**Ask anything. Upload docs. Get answers.**")
st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []

# --- Session Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- PDF Upload ---
uploaded_files = st.sidebar.file_uploader("📄 Upload PDF(s)", type="pdf", accept_multiple_files=True)
all_text = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            reader = PdfReader(tmp_file.name)
            for page in reader.pages:
                all_text += page.extract_text() or ""

# --- Main Title ---
st.title("💬 My Detox SPA - Agent")
st.markdown("### Powered by Cohere")

# --- Mic Section ---
with st.expander("🎙️ Voice Chat"):
    st.markdown("Speak your question and get a spoken response.")
    live_voice_chat.render_voice_ui()

# --- Text Input Chat ---
query = st.text_input("Type your question below:")
enable_audio = st.checkbox("🔊 Read answer aloud")

if st.button("Ask") and query:
    if all_text:
        prompt = f"""You are a helpful assistant answering based on the user's documents.

User question: {query}

Relevant document content:
{all_text}

Answer:"""
        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=400,
            temperature=0.3,
        )
        answer = response.generations[0].text.strip()
    else:
        answer = get_rag_answer(query)

    st.session_state.messages.append((query, answer))

# --- Chat History ---
for q, a in reversed(st.session_state.messages):
    with st.chat_message("user"):
        st.markdown(f"**You:** {q}")
    with st.chat_message("ai"):
        st.markdown(a)
        st.code(a, language="text")
        if enable_audio:
            tts = gTTS(text=a)
            tts.save("response.mp3")
            st.audio("response.mp3", autoplay=True)

# --- Transcript Download ---
if st.session_state.messages:
    chat_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.messages])
    st.download_button("📥 Download Transcript", data=chat_text, file_name="evolve_chat.txt")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 14px;'>
        <strong>Product Evolve</strong> |
        <a href='mailto:contact@productevolve.io'>contact@productevolve.io</a> |
        <a href='https://productevolve.io/' target='_blank'>productevolve.io</a>
    </div>
    """,
    unsafe_allow_html=True,
)

