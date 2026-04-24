import sys
from app.groq_service import analyze_behavior

features = {
    "UserId": "user-bot-login",
    "SessionId": "session-002",
    "CurrentPage": "/login",
    "Context": "preAuth",
    "AvgMouseSpeed": 2.5,
    "StdMouseSpeed": 0.01,
    "MouseMoveCount": 200,
    "AvgMouseIdle": 1,
    "AvgClickDuration": 10,
    "ClickCount": 40,
    "AvgClickInterval": 50,
    "AvgDwell": 2,
    "AvgFlight": 2,
    "KeyEventCount": 0,
    "TypingRate": 0,
    "ClickRate": 12,
    "MouseMoveRate": 20
}

try:
    res = analyze_behavior(confidence=0.6789, features=features)
    print("SUCCESS")
    print(res)
except Exception as e:
    print("FAILED")
    print(type(e).__name__)
    print(e)
