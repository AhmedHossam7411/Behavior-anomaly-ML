from tabpfn import TabPFNClassifier
import numpy as np
from app.db import get_data

class MLModel:
    def __init__(self):
        self.model = TabPFNClassifier()

        df = get_data()

        self.feature_names = [
            "AvgMouseSpeed",
            "StdMouseSpeed",
            "MouseMoveCount",
            "AvgMouseIdle",
            "AvgClickDuration",
            "ClickCount",
            "AvgClickInterval",
            "AvgDwell",
            "AvgFlight",
            "KeyEventCount",
            "TypingRate",
            "ClickRate",
            "MouseMoveRate"
        ]

        X = df[self.feature_names].values
        y = self.generate_labels(df)

        self.model.fit(X, y)

    def generate_labels(self, df):
        y = []

        for _, row in df.iterrows():
            if (
                row["TypingRate"] > df["TypingRate"].mean() * 1.5 or
                row["MouseMoveRate"] < df["MouseMoveRate"].mean() * 0.5 or
                row["ClickRate"] > df["ClickRate"].mean() * 2
            ):
                y.append(1)
            else:
                y.append(0)

        return np.array(y)

    # 🔥 UPDATED: safer + supports extra fields
    def _prepare_input(self, data):
        """
        Accepts:
        - list of lists (2D array)
        - list of dicts (object-based input)
        Ignores non-feature fields (UserId, SessionId, etc.)
        """

        # Case 1: already 2D array
        if isinstance(data, list) and isinstance(data[0], list):
            return np.array(data, dtype=float)

        # Case 2: list of objects
        if isinstance(data, list) and isinstance(data[0], dict):
            X = []

            for item in data:
                row = []

                for feature in self.feature_names:
                    value = item.get(feature, 0)

                    # 🔥 SAFE conversion (prevents crashes)
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        value = 0.0

                    row.append(value)

                X.append(row)

            return np.array(X, dtype=float)

    def predict(self, data):
        X = self._prepare_input(data)

        probs = self.model.predict_proba(X)
        results = []

        for i in range(len(probs)):
            anomaly_prob = float(probs[i][1])

            results.append({
                "confidence": anomaly_prob
            })

        return results