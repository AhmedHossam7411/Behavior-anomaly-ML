from tabpfn import TabPFNClassifier
import numpy as np

class MLModel:
    def __init__(self):
        self.model = TabPFNClassifier()

        X_train = np.array([
            [1,2,3,4],
            [4,3,2,1],
            [1,1,1,1],
            [2,2,2,2]
        ])
        y_train = np.array([0, 1, 0, 1])

        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)