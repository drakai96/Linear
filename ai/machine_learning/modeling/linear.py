from ai.machine_learning.modeling.base import BaseModelLinear, BasePredict
from ai.machine_learning.dataprecessing.preprocessing import PrecessingPredict
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pickle as pkl
import os
import time
import json


class LinearTraining(BaseModelLinear):
    def __init__(
        self, model_path, Xtrain: None, Xtest: None, ytrain: None, ytest: None
    ):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.model_path = f"ai/model_info/model_ai/{model_path}"
        self.model_linear = LinearRegression(fit_intercept=False)
        self.Xtrain_feature_selection, self.Xtest_feature_selection = (
            self.precessing_with_model()
        )

    def precessing_with_model(self):
        # Cov
        data = np.c_[self.ytrain, self.Xtrain]
        cov = np.cov(data.T)
        columns_important = np.abs(cov[1:, 0])
        top_20_value = sorted(columns_important, reverse=True)[20]
        index = np.where(columns_important > top_20_value)
        Xtrain_feature_selection = self.Xtrain[:, index[0].tolist()]
        Xtest_feature_selection = self.Xtest[:, index[0].tolist()]
        index = list(index[0])
        self.index = [int(i) for i in index]
        return Xtrain_feature_selection, Xtest_feature_selection

    def training(self):
        Xtrain_feature_selection, _ = self.precessing_with_model()
        Xtrain, ytrain = Xtrain_feature_selection, self.ytrain
        t1 = time.time()
        self.model_linear.fit(X=Xtrain, y=ytrain)
        t2 = time.time()
        self.time = t2 - t1
        self.coef_ = self.model_linear.coef_
        with open(self.model_path, "wb+") as f:
            pkl.dump(self.model_linear, f)
        return self.model_linear

    def eval_model(self, force_train=True):
        if os.path.exists(self.model_path) and not force_train:
            model_linear = self.load_model()
        else:
            model_linear = self.training()
        self.ypred = model_linear.predict(self.Xtest_feature_selection)
        mape = mean_absolute_percentage_error(self.ytest, self.ypred)
        mse = mean_squared_error(self.ytest, self.ypred)
        self.eval = {
            "mape": mape,
            "mse": mse,
            "name": "Linear",
            "training_time": self.time,
        }
        return self.eval, tuple(self.index)

    def load_model(self):
        if os.path.exists(self.model_path):
            return pkl.load(open(self.model_path, "rb+"))
        else:
            return self.training()


class LinearPredict(BasePredict):
    def __init__(self, data: pd.DataFrame, id: int):
        precessing_info = json.load(open("ai/data_info/json_data_info.json", "r"))
        self.data = data
        model_info = json.load(open("ai/model_info/model_info.json", "r"))

        self.precessing_info = precessing_info.get(str(id))
        self.model_info = model_info.get(str(id))
        self.impotant_varible = self.model_info.get("index_linear")
        self.id = id
        self.model = self.load_model()

    def load_model(self):
        model_path = self.model_info.get("model_path").get("linear")
        model = pkl.load(open(model_path, "rb+"))
        return model

    def precessing(self):
        precessing = PrecessingPredict(data=self.data, id=self.id)
        data_predict = precessing.data_use_predict()
        data = precessing.data
        return data_predict, data

    def predict(self):
        data_predict, data = self.precessing()
        data_predict = data_predict[:, self.impotant_varible]
        pred = self.model.predict(data_predict)
        pred = np.exp(pred)
        return data_predict, data
