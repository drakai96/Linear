from ai.machine_learning.dataprecessing.preprocessing import Precessing
from ai.machine_learning.modeling.linear import LinearTraining, LinearPredict
from ai.machine_learning.modeling.lasso import LassoTraining, LassoPredict
from ai.machine_learning.modeling.ridge import RidgeTraining, RidgePredict
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import json


class PipeLineTraining:
    def __init__(self, data_info: dict, id: int):
        self.id = id
        self.data_info = data_info
        self.data_path = data_info.get("data_path")
        self.target_name = data_info.get("target_name")
        self.data_precessing = data_info.get("data_precessing_path")
        self.model_info = 1

    def precessing(self):
        if self.data_precessing:
            all_data = np.load(self.data_precessing)
            X = all_data["data"]
            y = all_data["labels"]
            return train_test_split(X, y, test_size=0.2)
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path)
            Xtrain, Xtest, ytrain, ytest = Precessing(self.data_info).pipeline_runing()
            return Xtrain, Xtest, ytrain, ytest
        return None

    def modeling(self, force_training=True):
        all_data = self.precessing()
        if all_data is not None:
            Xtrain, Xtest, ytrain, ytest = all_data
            if self.model_info == 1:
                if not os.path.exists("ai/model_info/model_info.json"):
                    linear = LinearTraining(
                        f"model_linear_{self.id}.pkl", Xtrain, Xtest, ytrain, ytest
                    )
                    lasso = LassoTraining(
                        f"model_lasso_{self.id}.pkl", Xtrain, Xtest, ytrain, ytest
                    )
                    ridge = RidgeTraining(
                        f"model_ridge_{self.id}.pkl", Xtrain, Xtest, ytrain, ytest
                    )
                    linear_eval, list_index = linear.eval_model(force_training)
                    lasso_eval = lasso.eval_model(force_training)
                    ridge_eval = ridge.eval_model(force_training)
                    model_info = {}
                    model_info.update(
                        {
                            self.id: {
                                "model_path": {
                                    "linear": f"ai/model_info/model_ai/model_linear_{self.id}.pkl",
                                    "lasso": f"ai/model_info/model_ai/model_lasso_{self.id}.pkl",
                                    "ridge": f"ai/model_info/model_ai/model_ridge_{self.id}.pkl",
                                },
                                "index_linear": list_index,
                                "eval": [linear_eval, lasso_eval, ridge_eval],
                            }
                        }
                    )
                    with open("ai/model_info/model_info.json", "w") as f:
                        json.dump(model_info, f, ensure_ascii=True, indent=4)
                    return [linear_eval, lasso_eval, ridge_eval]
                else:
                    model_info = json.load(open("ai/model_info/model_info.json", "r"))
                    return model_info.get(str(self.id)).get("eval")
        else:
            return None


# if __name__ == "__main__":
#     data_dict = {"data_path": "ai/data_info/train.csv", "target_name": "SalePrice"}
#     ax = PipeLineTraining(data_info=data_dict).modeling()
#     print(ax)


class PipeLinePredict:
    def __init__(self, data: pd.DataFrame, id: int) -> None:
        self.data = data
        self.id = id

    def predict(self):
        lasso_predict, _ = LassoPredict(self.data, self.id).predict()
        ridge_predict, _ = RidgePredict(self.data, self.id).predict()
        linear_predict, _ = LinearPredict(self.data, self.id).predict()
        data = pd.DataFrame(_)
        data["lasso_predict"] = lasso_predict
        data["linear_predict"] = linear_predict
        data["ridge_pred"] = ridge_predict
        return data
