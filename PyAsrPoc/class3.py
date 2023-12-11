import gradio as gr

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
    #y = y.astype(np.float32)
    #y /= np.max(np.abs(y))

    audio_float32 = int2float(y)
    confidence = model(torch.from_numpy(audio_float32), sr).item()
    # if(is_speech):
    #     text1 = 'yes'
    # else : 
    #     text1 = 'no'

    return stream, confidence#8T8T8952({"sampling_rate": sr, "raw": stream})["text"]°3 06 °8790I?V  B.GLIHDVVSQVBNL/NTNG?NXS?CS.S.S.


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