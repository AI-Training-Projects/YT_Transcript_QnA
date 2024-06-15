"""
# YouTube Chatbot using LangChain and OpenAI
https://betterprogramming.pub/youtube-chatbot-using-langchain-and-openai-f8faa8f34929

Unlock the power to interact with YouTube videos from your command line
by Sami Maameri in Better Programming
Published in Better Programming
14 min read, Jun 18, 2023

"""

import streamlit as st
# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI  # Corrected import for ChatOpenAI
# from langchain.chains import LLMChain
# from huggingsound import SpeechRecognitionModel
# from pytube import YouTube
import subprocess
import os
import requests
import openai
import logging


# Create an instance of the YoutubeLoader: Instantiate the YoutubeLoader class by providing the YouTube video URL as a parameter. For example:

from langchain.document_loaders import YoutubeLoader

#VIDEO_ID_FULL = "https://www.youtube.com/watch?v=nZZQxHRoWX0"

# Create a Loader for the video transcript
VIDEO_ID = "nZZQxHRoWX0"
yt_url = f"https://www.youtube.com/watch?v={VIDEO_ID}"
loader = YoutubeLoader.from_youtube_url(yt_url)

# • Load and parse the transcript: Use the load() method of the YoutubeLoader instance to load and parse the transcript of the YouTube video. This method returns a Document object that represents the transcript.

video_transcript = loader.load()
print(f"Transcript for video {VIDEO_ID}: {video}")
print(f"type(video_transcript): {type(video_transcript)}")
# • Access the transcript content: Once the transcript is loaded, you can access its content through the content attribute of the Document object. This will give you the text of the transcript that you can further process or analyze.

transcript_text = video_transcript.content

# By following these steps, you can use the LangChain loader for YouTube text transcripts to easily retrieve and work with the transcript of a YouTube video.