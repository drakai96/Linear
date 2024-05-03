from ai.machine_learning.modeling.base import BaseModelLinear, BasePredict
from ai.machine_learning.dataprecessing.preprocessing import PrecessingPredict
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pickle as pkl
import os
import json
import pandas as pd
import time
import numpy as np


class RidgeTraining(BaseModelLinear):
    def __init__(
        self, model_path, Xtrain: None, Xtest: None, ytrain: None, ytest: None
    ):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.model_path = f"ai/model_info/model_ai/{model_path}"
        self.model_ridge = RidgeCV(
            fit_intercept=False, alphas=(0.001, 0.1, 0.5, 1, 2, 10, 100), cv=5
        )
        self.Xtrain_feature_selection, self.Xtest_feature_selection = (
            self.precessing_with_model()
        )

    # Ridge auto shrink the parameter so we don't need to feature selection
    def precessing_with_model(self):
        # Cov
        Xtrain_feature_selection = self.Xtrain
        Xtest_feature_selection = self.Xtest
        return Xtrain_feature_selection, Xtest_feature_selection

    def training(self):
        Xtrain, ytrain = self.Xtrain_feature_selection, self.ytrain
        t1 = time.time()
        self.model_ridge.fit(X=Xtrain, y=ytrain)
        self.time = time.time() - t1
        self.coef_ = self.model_ridge.coef_
        with open(self.model_path, "wb+") as f:
            pkl.dump(self.model_ridge, f)
        return self.model_ridge

    def eval_model(self, force_train=True):
        if os.path.exists(self.model_path) and not force_train:
            model_ridge = self.load_model()
        else:
            model_ridge = self.training()
        self.ypred = model_ridge.predict(self.Xtest)
        mape = mean_absolute_percentage_error(self.ytest, self.ypred)
        mse = mean_squared_error(self.ytest, self.ypred)
        self.eval = {
            "mape": mape,
            "mse": mse,
            "name": "Ridge",
            "training_time": self.time,
        }
        return self.eval

    def load_model(self):
        if os.path.exists(self.model_path):
            return pkl.load(open(self.model_path, "rb+"))
        else:
            return self.training()


class RidgePredict(BasePredict):
    def __init__(self, data: pd.DataFrame, id: int):
        precessing_info = json.load(open("ai/data_info/json_data_info.json", "r"))
        self.data = data
        model_info = json.load(open("ai/model_info/model_info.json", "r"))
        self.precessing_info = precessing_info.get(str(id))
        self.model_info = model_info.get(str(id))
        self.id = id
        self.model = self.load_model()

    def load_model(self):
        model_path = self.model_info.get("model_path").get("ridge")
        model = pkl.load(open(model_path, "rb+"))
        return model

    def precessing(self):
        precessing = PrecessingPredict(data=self.data, id=self.id)
        data_predict = precessing.data_use_predict()
        data = precessing.data
        return data_predict, data

    def predict(self):
        data_predict, data = self.precessing()
        pred = self.model.predict(data_predict)
        pred = np.exp(pred)
        return pred, data
