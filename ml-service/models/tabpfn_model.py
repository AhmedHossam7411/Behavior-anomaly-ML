from tabpfn import TabPFNClassifier
import numpy as np
from sklearn.model_selection import train_test_split
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
            "HackingStringDetected",
            "PasteCount",
            "SuspiciousPasteDetected",
            "DevToolsShortcutCount",
            "AbnormalInputDetected",
            "DevToolsDetected",
            "UnauthorizedAttempts",
        ]

        X = df[self.feature_names].values
        y = self.generate_labels(df)

        if len(X) > 50:
            X, _, y, _ = train_test_split(X, y, train_size=50, stratify=y, random_state=42)

        self.model.fit(X, y)

    def generate_labels(self, df):
        y = []
        for _, row in df.iterrows():
            hacking   = row.get("HackingStringDetected", 0) == 1
            sus_paste = row.get("SuspiciousPasteDetected", 0) == 1
            abnormal  = row.get("AbnormalInputDetected", 0) == 1
            unauth    = row.get("UnauthorizedAttempts", 0) > 2
            dt_probe  = (row.get("DevToolsDetected", 0) == 1 or
                         row.get("DevToolsShortcutCount", 0) >= 1)
            robotic   = (row.get("StdMouseSpeed", 999) < 0.05 and
                         row.get("TypingRate", 999) == 0.0 and
                         row.get("ClickRate", 0) > 5)
            y.append(1 if (hacking or sus_paste or abnormal or unauth or dt_probe or robotic) else 0)
        return np.array(y)

    def _get(self, item, *keys, default=0):
        for k in keys:
            v = item.get(k)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return float(default)

    def _hard_rules_floor(self, item):
        hacking   = self._get(item, "HackingStringDetected",  "hackingStringDetected")
        sus_paste = self._get(item, "SuspiciousPasteDetected","suspiciousPasteDetected")
        abnormal  = self._get(item, "AbnormalInputDetected",  "abnormalInputDetected")
        unauth    = self._get(item, "UnauthorizedAttempts",   "unauthorizedAttempts")
        dt_count  = self._get(item, "DevToolsShortcutCount",  "devToolsShortcutCount")
        dt_open   = self._get(item, "DevToolsDetected",       "devToolsDetected")
        std_mouse = self._get(item, "StdMouseSpeed",          "stdMouseSpeed", default=999)
        typing    = self._get(item, "TypingRate",             "typingRate",    default=999)
        click_r   = self._get(item, "ClickRate",              "clickRate",     default=0)

        if hacking == 1 or sus_paste == 1 or abnormal == 1 or unauth > 2:
            return 0.88

        if dt_open == 1 or dt_count >= 1:
            return 0.78

        if std_mouse < 0.05 and typing == 0.0 and click_r > 5:
            return 0.82

        return 0.0

    def _prepare_input(self, data):
        if isinstance(data, list) and isinstance(data[0], list):
            return np.array(data, dtype=float)

        if isinstance(data, list) and isinstance(data[0], dict):
            X = []
            for item in data:
                row = []
                for feature in self.feature_names:
                    camel = feature[0].lower() + feature[1:]
                    value = item.get(feature)
                    if value is None:
                        value = item.get(camel, 0)
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
            tabpfn_prob = float(probs[i][1])

            if isinstance(data, list) and isinstance(data[i], dict):
                floor = self._hard_rules_floor(data[i])
                tabpfn_prob = max(tabpfn_prob, floor)

            results.append({"confidence": tabpfn_prob})

        return results
