from pydantic import BaseModel
from typing import List


class DataInfo(BaseModel):
    feature_usefull: List[str, str]
    categorical_name: List[str, str]
    int_name: List[str, str]
    numerical_name: List[str, str]
    float_name: List[str, str]
    onehot_path: str = "onehot.pkl"
    onehot_path: str = ("onehot.pkl",)
    minmax_scale_path: str = ("minmaxscale.pkl",)
    target_flag: bool = (True,)
    target_name: str = ("SalePrice",)
    encode_path: str = ("ai/data_info/tranform_fit/onehot.pkl",)
    scaler_path: str = "ai/data_info/tranform_fit/minmaxscale.pkl"
