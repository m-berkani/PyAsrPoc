import gradio as gr
from transformers import pipeline
import numpy as np
import webrtcvad
import torch
import json
import requests

from Client import online
#transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound



def on_streamOrUpload(stream, examType,new_chunk, uploaded_file_path):
    global text
    global file_count
    
    #Mode Offline
    if (uploaded_file_path is not None):
        text=offline_mode(uploaded_file_path,examType)
        uploaded_file_path=None
        
    #Mode Online
    if(new_chunk is not None):
        sr, y = new_chunk
        #stream, text= online_mode(stream,new_chunk)
        audio_float32 = int2float(y)
        confidence = model(torch.from_numpy(audio_float32), sr).item()
        if confidence> vad_threshold:
            if stream is not None :
                stream = np.concatenate([stream, audio_float32])
            else:
                stream = audio_float32
        else:
            if (stream is not None):
                tensor = torch.tensor(np.concatenate([stream, audio_float32]))
                file_path = str(file_count)+".wav"
                save_audio(file_path,tensor,sr)
                text=offline_mode(file_path,examType)
                stream=None
                file_count=file_count+1
    return stream, text

def online_mode(stream,new_chunk):
    sr, y = new_chunk
    audio_float32 = int2float(y)
    confidence = model(torch.from_numpy(audio_float32), sr).item()
    transcription_text2=""
    global file_count
    if confidence> vad_threshold:
        if stream is not None :
            stream = np.concatenate([stream, audio_float32])
        else:
            stream = audio_float32
    else:
        if (stream is not None):
            tensor = torch.tensor(np.concatenate([stream, audio_float32]))
            file_path = str(file_count)+".wav"
            save_audio(file_path,tensor,sr)
            transcription_text2 = transcribe_wav_file(file_path)
            stream=None
            file_count=file_count+1
    return stream, transcription_text2 

def transcribe_wav_file(file_path,examType):
    url = "http://65.21.200.122:30000/uploadfile/"
    with open(file_path, "rb") as file:
        data = {"examType": examType}
        files = {'fileUpload': (file_path, file, 'audio/wav')}
        response = requests.post(url, files=files, data=data)
    
    print(response.json())
    return response.json()

def offline_mode(audio_path,examType):
    
    # Generate VAD timestamps
    transcription_text=""
    VAD_SR = 16000
    wav = read_audio(audio_path, sampling_rate=VAD_SR)
    t = get_speech_timestamps(wav, model, sampling_rate=VAD_SR, threshold=vad_threshold)

    
    # Add a bit of padding, and remove small gaps
    for i in range(len(t)):
        t[i]["start"] = max(0, t[i]["start"] - 3200)  # 0.2s head
        t[i]["end"] = min(wav.shape[0] - 16, t[i]["end"] + 20800)  # 1.3s tail
        if i > 0 and t[i]["start"] < t[i - 1]["end"]:
            t[i]["start"] = t[i - 1]["end"]  # Remove overlap
    

    temp_path="temp.wav"
    # Merge speech chunks
    
    save_audio(
        temp_path,
        collect_chunks(t, wav),
        sampling_rate=VAD_SR,
    )

    ##upload to API for transcription
    transcription_text = transcribe_wav_file(temp_path,examType)  
    return transcription_text

# Function to fetch a list of exams from the given URL
def get_exam_types():
    url = "http://65.21.200.122:30000/getExamType/"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            exams_string = response.text.strip('[]').replace(' ', '')  
            exam_types = exams_string.split(',')
            return exam_types
        else:
            print(f"Failed to fetch exam types. Status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

vad_threshold = 0.5
model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
file_count = 1
text =""
exam_types = get_exam_types()
demo = gr.Interface(
    on_streamOrUpload,
    ["state",gr.Dropdown(choices=exam_types, label="Exam Type", info="Select exam type"), gr.Audio(sources=[ "microphone"], streaming=True),gr.File(label="Upload File")],
    ["state", "text"],
    live=True,
)

demo.launch()