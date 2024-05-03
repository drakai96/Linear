# from fastapi import FastAPI, UploadFile, File
# from pydantic import BaseModel
# from fastapi.responses import JSONResponse
# import pandas as pd
# import numpy as np
# import os
# from ai.machine_learning.dataprecessing.preprocessing import Precessing,PrecessingPredict
# from ai.machine_learning.modeling.ridge import RidgeCV
# from ai.machine_learning.pipe_line import PipeLineTraining
# import json
# app = FastAPI()


# @app.get("/checkheath", description="check heath sever", tags=["Check heath"])
# def check_heath():
#     return "Alive!"


# @app.get("/categorical_visualization")
# def categorical_visual():
#     data = pd.DataFrame("data/train.csv")
#     value = data.groupby(by='KitchenQual').count().iloc[:,0].to_dict()
#     return value

# @app.post("/upload", description="Train data upload", tags=["Post data"])
# def upload_data(
#     target_name: str,
#     file: UploadFile = File(...),
# ):

#     data = pd.read_csv(file.file)

#     return {"Upload success"}

# @app.get("/precessing_data")
# def precessing_data(path = "data/train.csv",target_name="SalePrice"):
#     if os.path.exists(path):
#         Precessing(data = path,target_name=target_name)
#         return {"mesage": "success"}

# @app.get("/training")
# def training(id_data):
#     path = f"data/datatrain{id_data}.npz"
#     if os.path.exists(path = path):
#         input_param = json.load(open(path,"rb+"))
#         if os.path.exists(path= path):
#             data = np.load(path)
#             X = data["data"]
#             y = data["labels"]

#     else:
#         return JSONResponse(status_code=400, content={"status": False,"message": "Data is not exist"})

# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run("main:app", reload=True)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import api_router

origins = ["*"]
app = FastAPI(title="NewsAdvise")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", reload=True, host="localhost")
