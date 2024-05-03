from fastapi import APIRouter, UploadFile, File
import json
from ai.machine_learning.pipe_line import PipeLinePredict
import pandas as pd

# from be.api.schemas import DataInfo
import sys
import os

sys.path.append("")
if not os.path.exists("ai/data_info/json_data_info.json"):
    db = {}
else:
    db = json.load(open("ai/data_info/json_data_info.json", "r"))

router = APIRouter()


@router.post("/prediction/{id}")
def prediction(id: int = 1, file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    resposne = PipeLinePredict(data=data, id=id).predict()
    return resposne
