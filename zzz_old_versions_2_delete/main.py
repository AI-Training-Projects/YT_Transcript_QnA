"""
Old version that uses
streamlit 0.89.0 
openai v0.27.0
"""

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain import OpenAI, LLMChain
from langchain_community import LLMChain
#from langchain_core import OpenAI, LLMChain
#from langchain_experimental import OpenAI, LLMChain
from huggingsound import SpeechRecognitionModel
from pytube import YouTube
import torch
import librosa
import soundfile as sf
import subprocess
import os
import requests
import openai

API_URL = "https://api-inference.huggingface.co/models/jonatasgrosman/wav2vec2-large-xlsr-53-english"
headers = {"Authorization": "Bearer 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"}
st.set_page_config(
    page_title="YT Video Chatbot",
    page_icon=":video_camera:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json().get('text', '')


def story(text,text2, api):
    template = """
    You are youtube video chatbot.
    You need to answer users from the given transcript. 
    The answer should be accurate according to the given transcript
    transcript: {text}
    question : {text2}
"""
    prompt = PromptTemplate(template=template, input_variables=["text","text2"])
    llm_model = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo",
                         temperature=1, openai_api_key=api), prompt=prompt, verbose=True)
    ans = llm_model.predict(text=text,text2=text2)
    return ans


def transcribe_audio(URL):
    yt = YouTube(URL)
    yt.streams \
        .filter(only_audio=True, file_extension='mp4') \
        .first() \
        .download(filename='ytaudio.mp4')
        
    # Convert the audio to WAV format
    ffmpeg_path = os.path.join(os.getcwd(), 'ffmpeg', 'bin', 'ffmpeg.exe')
    subprocess.call(f'{ffmpeg_path} -i ytaudio.mp4 -acodec pcm_s16le -ar 16000 ytaudio.wav -y', shell=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the ASR model & Audio Chunking
    input_file = 'ytaudio.wav'
    print(librosa.get_samplerate(input_file))
    
    # Stream over 30 seconds chunks rather than load the full file
    stream = librosa.stream(
        input_file,
        block_length=30,
        frame_length=16000,
        hop_length=16000
    )
    os.makedirs('content', exist_ok=True)

    for i, speech in enumerate(stream):
        file_path = f'content/{i}.wav'
        sf.write(file_path, speech, 16000)
    audio_path = []
    
    for a in range(i + 1):
        audio_path.append(f'content/{a}.wav')
    full_transcript = ' '
    
    for a in range(i):
        #full_transcript += query(f'content/{a}.wav')
        full_transcript+=query(f'content/{a}.wav') #["text"]     # Edited out to fix error
        return full_transcript


def main():
    st.header("📹 Chat with YT video 💬")
    URL = st.text_input('Enter YouTube video URL')
    if URL:
        with st.spinner('Wait for it...'):
            text = transcribe_audio(URL)
            st.balloons()

        text2 = st.text_input("enter your question")
        if text2 != "":
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                st.error('API key not found in environment variables.')
                return
            with st.spinner('Wait for it...'):
                ans = story(text=text, text2=text2)
                st.write(ans)
                st.balloons()
                
if __name__ == '__main__':
    main()
