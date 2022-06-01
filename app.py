from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile , Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import re
import sys
import os
import base64
import pandas as pd
from load import *
import boto3
import config
from PIL import Image

sys.path.append(os.path.abspath("./model"))

wbook = 'E47grading.xlsx' # 엑셀파일 저장 경로

class Inputs(BaseModel):
    AnswerId: str
    name: str
    studentID: str
    is_certified: int

class InputRequest(BaseModel):
    inputs: list

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


origins = [
    "http://localhost:8001",
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client_s3 = boto3.client(
        's3',
        region_name = config.AWS_S3_CONFIG['AWS_REGION'],
        aws_access_key_id = config.AWS_S3_CONFIG['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key = config.AWS_S3_CONFIG['AWS_SECRET_ACCESS_KEY']
)

@app.post('/auto-score')
async def auto_score(inputs:InputRequest):
  try:
    data = inputs.dict()
    students = pd.DataFrame(data['inputs'])
    file_name= "static/"+str(datetime.now()) + wbook
    main(students, file_name)
    print(wbook)
    return file_name
  except Exception as e:
    print('EXCEPTION: {}'.format(e))
    return e


