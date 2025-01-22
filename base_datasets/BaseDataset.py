from abc import ABC, abstractmethod

class BaseDataset(ABC):
    @abstractmethod
    def load_data(self):
        """Should return X, y"""
        pass