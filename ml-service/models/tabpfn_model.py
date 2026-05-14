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
            "MouseMoveRate",
            "HackingStringDetected",      # 1 = attack string typed/in URL
            "PasteCount",                 # pastes in this window
            "SuspiciousPasteDetected",    # 1 = pasted content matched attack pattern
            "DevToolsShortcutCount",      # F12/Ctrl+Shift+I/Ctrl+U presses
            "AbnormalInputDetected",      # 1 = input exceeded 500 chars
            "DevToolsDetected",           # 1 = window dimensions suggest DevTools open
            "UnauthorizedAttempts",       # challenge bypass navigation attempts
        ]

        X = df[self.feature_names].values
        y = self.generate_labels(df)

        self.model.fit(X, y)

    def generate_labels(self, df):
        y = []

        for _, row in df.iterrows():
            # Hard anomaly signals — any one of these = definitive malicious intent
            if (
                row.get("HackingStringDetected", 0) == 1 or
                row.get("SuspiciousPasteDetected", 0) == 1 or
                row.get("AbnormalInputDetected", 0) == 1 or
                row.get("UnauthorizedAttempts", 0) > 2
            ):
                y.append(1)
            # Soft anomaly signals — behavioral outliers or tool-use patterns
            elif (
                row["TypingRate"] > df["TypingRate"].mean() * 1.5 or
                row["MouseMoveRate"] < df["MouseMoveRate"].mean() * 0.5 or
                row["ClickRate"] > df["ClickRate"].mean() * 2 or
                row.get("DevToolsShortcutCount", 0) > 3 or
                (row.get("DevToolsDetected", 0) == 1 and row.get("DevToolsShortcutCount", 0) > 0)
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
                    camel_feature = feature[0].lower() + feature[1:]
                    
                    value = item.get(feature)
                    if value is None:
                        value = item.get(camel_feature, 0)

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