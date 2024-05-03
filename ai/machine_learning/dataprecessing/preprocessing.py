import pandas as pd
import numpy as np
import sweetviz as sw
import sklearn.impute as impute
from sklearn.preprocessing import MinMaxScaler
from ai.machine_learning.constant import Config
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from pydantic import BaseModel, Field
from typing import List
from sklearn.model_selection import train_test_split
import pickle as pkl
import json
import os


class ModelInfo(BaseModel):
    model_name: str = Field(description="Name of model training")
    model_path: str = Field(description="Directory of model")
    categorical_tranform: str = Field(
        default=None, description="Directory of model fitting"
    )
    numerical_tranform: str = Field(
        default=None, description="Directory of model fitting"
    )
    numerical_scaler: str = Field(default=None, description="Directory of model scaler")
    target_scaler: str = Field(default=False, description="Directory of model scaler")
    numerical_feature: List[str,] = Field(default=None)
    categorical_feature: List[str,] = Field(default=None)


class Precessing:
    def __init__(
        self,
        # file_path: str = None,
        data_info: dict,
        id,
    ):
        # if not file_path and not data:
        #     raise Exception("stupid input")
        data_all = pd.read_csv(data_info.get("data_path"))
        target_name = data_info.get("target_name")
        self.id = id
        data_all = data_all[data_all[target_name].notnull()]
        self.data = data_all.drop(columns=target_name)
        self.target_name = target_name
        self.target = data_all[target_name]
        self.pipeline_runing()
        self.precessing_parame()

    # @classmethod
    # def read_file(self, file_path: str, data) -> pd.DataFrame:
    #     if data:
    #         return self.data
    #     if file_path.endswith(".csv"):
    #         self.data = pd.read_csv(file_path)
    #     return self.data

    def define_categorical_numerical_data(self, is_visualization=False):
        categorical_feature = self.data.select_dtypes(include=["object"]).columns
        data_categorical = self.data[categorical_feature]
        numerical_feature = self.data.select_dtypes(
            include=["int64", "float64"]
        ).columns
        data_numeric = self.data[numerical_feature]
        print(data_categorical.info())
        print(data_numeric.info())
        if is_visualization:
            sw.analyze(self.data)
            os.rename("response.html", f"{self.id}.html")

    def handle_missing_value(self):
        data_copy = self.data.copy()
        analyst_columns = []
        for column in self.data.columns:
            feature = data_copy[column]
            if len(feature.dropna()) / len(data_copy) > 0.7:
                analyst_columns.append(column)
        self.feature_usefull = analyst_columns
        self.data = self.data[analyst_columns]
        categorical_feature = self.data.select_dtypes(include=["object"])
        self.categorical_name = categorical_feature.columns
        float_feature = self.data.select_dtypes(include=["float64"])
        self.float_name = float_feature.columns
        int_feature = self.data.select_dtypes(include=["int64"])
        self.int_name = int_feature.columns
        numerical_feature = self.data.select_dtypes(include=["int64", "float64"])
        self.numerical_name = numerical_feature.columns
        # impute Categorical value
        categorical_impute = impute.SimpleImputer(
            strategy=Config.categorical_impute_method
        )
        data_impute_categorical = categorical_impute.fit(categorical_feature)
        categorical_data = data_impute_categorical.transform(categorical_feature)
        categorical_data = pd.DataFrame(
            categorical_data, columns=categorical_feature.columns
        )

        # impute float value
        float_impute = impute.SimpleImputer(strategy=Config.numerical_impute_method)

        float_impute.fit(float_feature)
        float_data = float_impute.fit_transform(float_feature)
        float_data = pd.DataFrame(float_data, columns=float_feature.columns)

        # impute int value
        int_impute = impute.SimpleImputer(strategy=Config.categorical_impute_method)
        int_tranform_fit = int_impute.fit(int_feature)
        int_data = int_tranform_fit.fit_transform(int_feature)
        int_data = pd.DataFrame(int_data, columns=int_feature.columns)
        data_clean_missing_value = pd.concat(
            [categorical_data, float_data, int_data], axis=1
        )
        self.data_clean_missing_value = data_clean_missing_value
        self.categorical_feature = categorical_data.select_dtypes(include=["object"])
        self.float_feature = float_data.select_dtypes(include=["float64"])
        self.int_feature = int_data.select_dtypes(include=["int64"])
        self.numeric_feature = pd.concat([self.float_feature, self.int_feature], axis=1)
        return data_clean_missing_value

    def handle_target(self):

        # target_feature = self.target
        # skew = target_feature.skew()
        # if abs(skew) < 0.5:
        #     fig, ax = plt.subplots(figsize=(6, 20))
        #     ax.set_title(f"Histogram of pricesale, the histogram chart is skewness")
        # else:
        #     fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        #     pd.plotting.hist_series(target_feature, bins=50, ax=ax[0])
        #     ax[0].set_title("Normal Histogram")
        #     # Add your second plot here if needed
        #     pd.plotting.hist_series(np.log(target_feature), bins=50, ax=ax[1])
        #     ax[1].set_title("Normal scaling")
        # plt.show()
        pass

    def feature_engine(self):
        # log
        data = self.data_clean_missing_value.copy()
        self.target_flag = None
        skew = self.target.skew()
        if abs(skew) > 0.5:
            self.target = pd.Series(np.log(self.target), name=self.target.name)
            self.target_flag = True
        # Product and square feature
        self.engine = PolynomialFeatures(degree=2, include_bias=False)
        self.engine.fit(data[self.numerical_name])
        with open(f"ai/data_info/tranform_fit/poly_{self.id}.pkl", "wb+") as f:
            pkl.dump(self.engine, f)
        self.feature_engine_numerical = self.engine.transform(data[self.numerical_name])

    def tranform_data(self):
        # Categorical tranform
        encode_path = f"ai/data_info/tranform_fit/onehot_{self.id}.pkl"
        scale_path = f"ai/data_info/tranform_fit/minmaxscale_{self.id}.pkl"
        self.encode_path = encode_path
        self.scale_path = scale_path
        categorical_tranform = OneHotEncoder(drop="first")
        categorical_tranform.fit(self.data[self.categorical_name])
        with open(encode_path, "wb+") as f:
            pkl.dump(categorical_tranform, f)

        self.feature_engine_categorical = categorical_tranform.transform(
            self.data[self.categorical_name]
        ).toarray()
        # numerical tranform
        scaler_feature_engine = MinMaxScaler(feature_range=(0, 1))
        self.scaler_feature_engine = scaler_feature_engine.fit(
            self.feature_engine_numerical
        )
        with open(scale_path, "wb+") as f:
            pkl.dump(self.scaler_feature_engine, file=f)

        self.feature_engine_numerical_tranform = self.scaler_feature_engine.transform(
            self.feature_engine_numerical
        )
        return self.feature_engine_categorical, self.feature_engine_numerical_tranform

    def pipeline_runing(self):
        print("define_categorical_numerical_data")
        self.define_categorical_numerical_data()
        print("handle_missing_value")
        self.handle_missing_value()
        print("handle_target")
        self.handle_target()
        print("feature_engine")
        self.feature_engine()
        print("tranform_data")
        self.tranform_data()
        X = np.c_[
            self.feature_engine_categorical,
            self.feature_engine_numerical_tranform,
        ]
        y = self.target
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.2, random_state=10, shuffle=True
        )
        data = pd.DataFrame(Xtrain)
        data.to_csv(f"ai/data_info/data_preprocessed/{self.id}.csv", index=False)
        np.savez(f"ai/data_info/data_preprocessed/{self.id}.npz", data=X, labels=y)
        return Xtrain, Xtest, ytrain, ytest

    def precessing_parame(self, path="ai/data_info/json_data_info.json") -> dict:
        path = f"ai/data_info/json_data_info.json"
        import json

        data = {}
        if os.path.exists(path=path):
            data = json.load(open("ai/data_info/json_data_info.json", "r"))
        preprocessing_info = {
            "data_path": f"ai/data_info/data_input/{self.id}.csv",
            "feature_usefull": list(self.feature_usefull),
            "categorical_name": list(self.categorical_name),
            "int_name": list(self.int_name),
            "numerical_name": list(self.numerical_name),
            "float_name": list(self.float_name),
            "poly_path": f"ai/data_info/tranform_fit/poly_{self.id}.pkl",
            "onehot_path": f"ai/data_info/tranform_fit/onehot_{self.id}.pkl",
            "minmax_scale_path": f"ai/data_info/tranform_fit/minmaxscale_{self.id}.pkl",
            "target_flag": self.target_flag,
            "target_name": self.target_name,
            "encode_path": self.encode_path,
            "scaler_path": self.scale_path,
            "data_precessing_path": f"ai/data_info/data_preprocessed/{self.id}.npz",
        }
        data.update({self.id: preprocessing_info})
        with open(path, "w+") as f:
            json.dump(data, f, indent=4, ensure_ascii=True)
        return preprocessing_info


class PrecessingPredict:
    def __init__(self, data: pd.DataFrame, id: int):
        self.id = id
        precessing_parame = json.load(
            open("ai/data_info/json_data_info.json", "r")
        ).get(str(id))
        model_param = json.load(open("ai/model_info/model_info.json", "r")).get(str(id))
        feature_use = precessing_parame.get("feature_usefull")
        self.data = data[feature_use]
        self.target_name = precessing_parame.get("target_name")
        self.numerical_name = precessing_parame.get("numerical_name")
        self.categorical_name = precessing_parame.get("categorical_name")
        self.int_name = precessing_parame.get("int_name")
        self.float_name = precessing_parame.get("float_name")
        self.minmax_path = precessing_parame.get("minmax_scale_path")
        self.scale_path = precessing_parame.get("scale_path")
        self.encode_categorical = precessing_parame.get("encode_path")
        self.fag_skew = precessing_parame.get("target_flag")
        self.onehot_path = precessing_parame.get("onehot_path")
        self.poly_path = precessing_parame.get("poly_path")
        self.feature_engine()

    def define_categorical_numerical_data(self, is_visualization=False):
        pass

    def handle_missing_value(self):
        self.data = self.data.dropna()
        return self.data

    def feature_engine(self):
        # Product and square feature
        self.handle_missing_value()
        self.engine = pkl.load(open(self.poly_path, "rb+"))
        self.feature_engine_numerical = self.engine.transform(
            self.data[self.numerical_name]
        )

    def tranform_data(self):
        # Categorical tranform
        with open(self.onehot_path, "rb+") as f:
            categorical_tranform = pkl.load(f)

        self.feature_engine_categorical = categorical_tranform.transform(
            self.data[self.categorical_name]
        ).toarray()
        # numerical tranform
        with open(self.minmax_path, "rb+") as f:
            scaler_feature_engine = pkl.load(f)

        self.feature_engine_numerical_tranform = scaler_feature_engine.transform(
            self.feature_engine_numerical
        )
        self.data_predict = np.c_[
            self.feature_engine_categorical, self.feature_engine_numerical_tranform
        ]
        return (
            self.data_predict,
            self.feature_engine_categorical,
            self.feature_engine_numerical_tranform,
        )

    def data_use_predict(self):
        data_predict, feature_engine_categorical, feature_engine_numerical_tranform = (
            self.tranform_data()
        )
        return data_predict


# if __name__ == "__main__":
#     data = pd.read_csv("train.csv").iloc[:, 1:]
#     precessing = Precessing(data_all=data, target_name="SalePrice").pipeline_runing()
#     # precessing.defind_categorical_numerical_data(is_visualization = True)
#     parame_input = json.load(
#         open("ai/data_info/data_preprocessed/preprocessing_info.json", "r")
#     )
#     data2 = PrecessingPredict(data=data, precessing_parame=parame_input)
#     data2.data_use_predict()
