from fastapi.responses import JSONResponse
from fastapi import APIRouter
import pandas as pd
import sys
import json

sys.path.append("")

router = APIRouter()


@router.get("/datapoint/{id}")
def datapoint(id: int = 1):
    data = pd.read_csv(f"ai/data_info/data_input/{id}.csv")
    bar_chart = data.groupby(by="Neighborhood").count()
    respone = [
        {"label": str(bar_chart.index[i]), "y": int(bar_chart.iloc[i, 0])}
        for i in range(len(bar_chart))
    ]
    return JSONResponse(status_code=200, content=respone)


@router.get("/linechart/{id}")
def line_chart(id: int = 1):
    data = pd.read_csv(f"ai/data_info/data_input/{id}.csv")
    bar_chart = data.groupby(by="YrSold").count()
    respone = [
        {"label": str(bar_chart.index[i]), "y": int(bar_chart.iloc[i, 0])}
        for i in range(len(bar_chart))
    ]
    return JSONResponse(status_code=200, content=respone)


@router.get("/lineperyear/{id}")
def line_per_year(id: int = 1):
    data = pd.read_csv(f"ai/data_info/data_input/{id}.csv")
    bar_chart = data.groupby(by="YrSold")["SalePrice"].mean()
    respone = [
        {"label": str(bar_chart.index[i]), "y": int(bar_chart.iloc[i])}
        for i in range(len(bar_chart))
    ]
    return JSONResponse(status_code=200, content=respone)


@router.get("/scatter/{id}")
def scatter_chart(id: int = 1):
    data = pd.read_csv(f"ai/data_info/data_input/{id}.csv")
    chart_value = data[["SalePrice", "LotArea"]]
    chart_value.columns = ["x", "y"]
    chart_data = json.loads(chart_value.to_json(orient="records"))
    return JSONResponse(status_code=200, content=chart_data)
