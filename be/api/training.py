from fastapi import APIRouter
import json
from ai.machine_learning.pipe_line import PipeLineTraining

# from be.api.schemas import DataInfo
import sys
import os

sys.path.append("")
if not os.path.exists("ai/data_info/json_data_info.json"):
    db = {}
else:
    db = json.load(open("ai/data_info/json_data_info.json", "r"))

router = APIRouter()


@router.get("/training/{id}")
def training(id: int = 1):
    input_parame = db.get(str(id))
    resposne = PipeLineTraining(input_parame, id).modeling()
    return resposne
