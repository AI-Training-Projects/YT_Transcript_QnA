import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from huggingsound import SpeechRecognitionModel
from pytube import YouTube
import torch
import librosa
import soundfile as sf
import subprocess
import os
import requests
import openai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
headers = {"Authorization": "Bearer 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"}
st.set_page_config(
    page_title="YT Video Chatbot",
    page_icon=":video_camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    text = response.json().get('text', '')
    logging.info(f"Transcribed text from {filename}: {text}")
    return text

def story(text, text2, api):
    template = """
    You are a YouTube video chatbot.
    You need to answer users from the given transcript.
    The answer should be accurate according to the given transcript.
    Transcript: {text}
    Question: {text2}
    """
    prompt = PromptTemplate(template=template, input_variables=["text", "text2"])
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1, openai_api_key=api)
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    ans = llm_chain.predict(text=text, text2=text2)
    logging.info(f"Generated response: {ans}")
    return ans

@st.cache_data
def download_audio(URL):
    yt = YouTube(URL)
    yt.streams.filter(only_audio=True, file_extension='mp4').first().download(filename='ytaudio.mp4')
    logging.info(f"Downloaded audio from {URL}")
    return 'ytaudio.mp4'

@st.cache_data
def convert_audio_to_wav(mp4_filename):
    wav_filename = 'ytaudio.wav'
    ffmpeg_path = os.path.join(os.getcwd(), 'ffmpeg', 'bin', 'ffmpeg.exe')
    subprocess.call(f'{ffmpeg_path} -i {mp4_filename} -acodec pcm_s16le -ar 16000 {wav_filename} -y', shell=True)
    logging.info(f"Converted {mp4_filename} to {wav_filename}")
    return wav_filename

@st.cache_data
def transcribe_audio(URL):
    mp4_filename = download_audio(URL)
    wav_filename = convert_audio_to_wav(mp4_filename)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_file = wav_filename
    logging.info(f"Starting transcription of {input_file}")
    stream = librosa.stream(input_file, block_length=30, frame_length=16000, hop_length=16000)
    os.makedirs('content', exist_ok=True)
    
    for i, speech in enumerate(stream):
        file_path = f'content/{i}.wav'
        sf.write(file_path, speech, 16000)
    audio_path = [f'content/{a}.wav' for a in range(i + 1)]
    full_transcript = ''
    
    for a in range(i):
        full_transcript += query(f'content/{a}.wav')
    logging.info(f"Full transcript: {full_transcript}")
    return full_transcript

def main():
    st.header("📹 Chat with YT video 💬")
    URL = st.text_input('Enter YouTube video URL')
    if URL:
        with st.spinner('Wait for it...'):
            text = transcribe_audio(URL)
            st.balloons()

        text2 = st.text_input("Enter your question")
        if text2:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                st.error('API key not found in environment variables.')
                return
            with st.spinner('Wait for it...'):
                ans = story(text=text, text2=text2, api=api_key)
                st.write(ans)
                st.balloons()

if __name__ == '__main__':
    main()
