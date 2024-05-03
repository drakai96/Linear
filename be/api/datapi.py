from fastapi import APIRouter, UploadFile, File
import pandas as pd
import json
import sys
import os
from fastapi.responses import HTMLResponse
from ai.machine_learning.dataprecessing.preprocessing import Precessing

sys.path.append("")


router = APIRouter()


@router.post("/data_upload/{id}")
def upload_data(id: int = 1, file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    if os.path.exists(f"ai/data_info/data_input/{id}.csv"):
        HTMLResponse(status_code=400, content="Name id is exit")
    data.to_csv(f"ai/data_info/data_input/{id}.csv", index=False)
    return HTMLResponse(status_code=200, content="success")


@router.get("/data/{id}")
def get_data(id: int = 1):
    data = pd.read_csv(f"ai/data_info/data_input/{id}.csv")[:100]
    if not os.path.exists(f"ai/data_info/data_input/{id}.csv"):
        HTMLResponse(status_code=400, content="Name id is not exit")
    response = json.loads(data.to_json(orient="records"))
    return response


@router.get("/precessing_data/{id}")
def precessing_data(target_name: str, id: int = 1):
    data_info = {
        "data_path": f"ai/data_info/data_input/{id}.csv",
        "target_name": target_name,
    }
    if os.path.exists(f"ai/data_info/data_input/{id}.csv"):
        Precessing(data_info=data_info, id=id)
    data = pd.read_csv(f"ai/data_info/data_preprocessed/{id}.csv")[:100]
    if not os.path.exists(f"ai/data_info/data_input/{id}.csv"):
        HTMLResponse(status_code=400, content="Name id is not exit")
    response = json.loads(data.to_json(orient="records"))
    return response
