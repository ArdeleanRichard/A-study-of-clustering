from abc import ABC, abstractmethod

class BaseClusteringAlgorithm(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass