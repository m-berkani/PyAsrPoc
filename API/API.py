from fastapi import FastAPI, File, UploadFile, Form
from aiofiles import open as async_open
import whisper
from typing import List
from whisper.tokenizer import LANGUAGES
from typing import Any, Optional
import uvicorn
import numpy as np
import db_config as db
from pydantic import BaseModel
    
class ArrayInput(BaseModel):
    data: List[List[float]]

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

#Upload a file and return filename as reponse
@app.post("/uploadfile3/")
async def uploadfile3(arr_input: ArrayInput):
    numpy_array = np.array(arr_input.data)
    # call whisper endpoint
    # vocabs=await db.get_Vocabs(ExamTypeName)
    # model_size = "small"
    # whisperModel = whisper.load_model(model_size)
    # date  = fileUpload.file.read()
    # audio_int16 = np.frombuffer(date, np.int16);
    # audio_float32 = int2float(audio_int16)
    # result = whisperModel.transcribe(audio_float32)
    # print(result)
    return {"received_array": numpy_array.tolist()}


#Upload a file and return filename as reponse
@app.post("/uploadfile")
async def uploadfile(fileUpload: UploadFile = File(...)):
    # call whisper endpoint
    ##vocabs=await db.get_Vocabs(ExamTypeName)
    global whisperModel
    date  = fileUpload.file.read()
    audio_int16 = np.frombuffer(date, np.int16);
    audio_float32 = int2float(audio_int16)
    result = whisperModel.transcribe(audio_float32)
    print(result)
    return {"transcription": result["text"],
        "Status": "200 ok"}


@app.get("/getExamType/")
async def getExamType():
    list_of_examTypeName= await db.get_list_examTypeName()
    
    return {list_of_examTypeName}

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()
    return sound

if __name__ == "__main__":
    model_size = "small"
    whisperModel = whisper.load_model(model_size)
    uvicorn.run(app, host="0.0.0.0", port=30000)