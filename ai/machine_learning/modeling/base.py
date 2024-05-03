from abc import ABC, abstractmethod


class BaseModelLinear(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def training(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def precessing_with_model(self):
        pass


class BasePredict(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def precessing(self):
        pass

    @abstractmethod
    def predict(self):
        pass
