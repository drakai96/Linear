from fastapi import APIRouter
from api import datapi
from api import visualization
from api import training
from api import prediciton

api_router = APIRouter()
api_router.include_router(datapi.router, prefix="/datapi", tags=["Data Upload"])
api_router.include_router(
    visualization.router, prefix="/visualization", tags=["Data Visualization"]
)
api_router.include_router(training.router, prefix="/training", tags=["Traning model"])
api_router.include_router(prediciton.router, prefix="/prediction", tags=["Prediction"])
