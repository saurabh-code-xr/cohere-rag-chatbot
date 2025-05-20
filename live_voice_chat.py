import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
from gtts import gTTS
import tempfile

# Load Whisper model
model = whisper.load_model("base")

# Global audio queue
audio_q = queue.Queue()

# Streamlit UI rendering for voice chat
def render_voice_ui():
    if "listening" not in st.session_state:
        st.session_state.listening = False

    if not st.session_state.listening:
        if st.button("🎤 Start Listening"):
            st.session_state.listening = True
            threading.Thread(target=listen_and_respond, daemon=True).start()
    else:
        st.success("Listening... Speak now!")
        if st.button("🛑 Stop Listening"):
            st.session_state.listening = False

# Audio callback
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

# Main logic to listen and respond
def listen_and_respond():
    with sd.InputStream(samplerate=16000, channels=1, callback=callback):
        audio_data = []

        while st.session_state.listening:
            try:
                data = audio_q.get(timeout=5)
                audio_data.append(data)
            except queue.Empty:
                break

        st.session_state.listening = False

        # Process audio
        audio_np = np.concatenate(audio_data, axis=0).flatten()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_filename = f.name
            sd.write(f, audio_np, 16000)

        result = model.transcribe(temp_filename)
        user_text = result["text"]

        st.info(f"**You said:** {user_text}")

        # Fake AI response for now (replace with real call)
        ai_response = f"You said: {user_text}"

        st.success(f"**AI:** {ai_response}")

        # Speak response
        tts = gTTS(text=ai_response)
        tts_path = "temp_response.mp3"
        tts.save(tts_path)
        st.audio(tts_path, autoplay=True)
