from fastapi import FastAPI, File, UploadFile
from aiofiles import open as async_open
import whisper
from whisper.tokenizer import LANGUAGES
from typing import Optional
import uvicorn
import numpy as np

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

#Upload a file and return filename as reponse
@app.post("/uploadfile")
async def uploadfile(fileUpload: UploadFile = File(...)):
    # call whisper endpoint
    model_size = "tiny"
    whisperModel = whisper.load_model(model_size)
    date  = fileUpload.file.read()
    audio_int16 = np.frombuffer(date, np.int16);
    audio_float32 = int2float(audio_int16)
    result = whisperModel.transcribe(audio_float32)
    return {"transcription": result["text"],
        "Status": "200 ok"}

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()
    return sound

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30000)