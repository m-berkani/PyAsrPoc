import gradio as gr
from transformers import pipeline
import numpy as np


import requests

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(stream, new_chunk):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    
    transcription = transcriber({"sampling_rate": sr, "raw": stream})["text"]
    return stream, transcription

def save_transcription_to_file(transcription, filename="transcription.txt"):
    with open(filename, "a") as file:
        file.write(transcription + "\n")

def save_and_transcribe(stream, new_chunk):
    stream, transcription = transcribe(stream, new_chunk)
    save_transcription_to_file(transcription)
    return stream, transcription

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

exam_types = get_exam_types()


demo = gr.Interface(
    save_and_transcribe,
    [
        "state",
        gr.Dropdown(["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"], label="Model", info="Select an option"),
        gr.Dropdown(choices=exam_types, label="Exam Type", info="Select exam type"),
        gr.Audio(sources=["microphone"], streaming=True),
        gr.File(label="Upload File", type="binary"),  # Use 'binary' instead of 'audio/*'
        gr.Button("Submit"),
    ],
    ["state", "text"],
    live=True,
)


demo.launch()