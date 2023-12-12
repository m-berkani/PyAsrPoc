import gradio as gr
from transformers import pipeline
import numpy as np
import webrtcvad
import torch
import json
import requests
#transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

def transcribe(stream, new_chunk):
    global file_count
    global text
    sr, y = new_chunk
    audio_float32 = int2float(y)
    confidence = model(torch.from_numpy(audio_float32), sr).item()
    if confidence> 0.5:
        if stream is not None :
            stream = np.concatenate([stream, audio_float32])
        else:
            stream = audio_float32
    else:
        if (stream is not None):
            tensor = torch.tensor(stream )
            save_audio(str(file_count)+".wav",tensor,sr)
            transcribe_api(str(file_count)+".wav")
            stream=None
            file_count=file_count+1
    return stream, confidence

def transcribe_api(file_path):
    url = "http://65.21.200.122:30000/uploadfile/"
    
    with open(file_path, "rb") as file:
        files = {'fileUpload': (file_path, file, 'audio/wav')}
        response = requests.post(url, files=files)
    
    print(response.json())
    return response.json()



model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
file_count = 1
text =""
demo = gr.Interface(
    transcribe,
    ["state", gr.Audio(sources=["microphone"], streaming=True)],
    ["state", "text"],
    live=True,
)

demo.launch()