from fastapi import APIRouter, Request
from pydantic import BaseModel
from app.save_prediction import save_prediction
from app.groq_service import analyze_behavior

router = APIRouter()


class PredictionRequest(BaseModel):
    data: list


# 🔥 Calibration function
def calibrate_confidence(prob):
    prob = prob * 0.7
    prob = max(0.05, min(0.95, prob))
    return float(prob)


@router.post("/predict")
def predict(request: PredictionRequest, req: Request):
    model = req.app.state.model
    data = request.data

    # 🔄 Unwrap nested .NET wrapper if present (e.g., [{'Data': [...]}] )
    if data and isinstance(data, list) and isinstance(data[0], dict):
        nested = data[0].get("Data") or data[0].get("data")
        if isinstance(nested, list) and len(nested) > 0:
            data = nested

    # 🔒 validation
    if not data or not isinstance(data, list):
        return {"error": "Invalid input"}

    # 🔥 ML prediction (RAW)
    result = model.predict(data)[0]


    # 🔥 IMPORTANT: calibrate confidence
    raw_confidence = result["confidence"]
    confidence = calibrate_confidence(raw_confidence)


    # 🔥 metadata
    user_id = None
    session_id = None
    features = {}

    if isinstance(data[0], dict):
        user_id = data[0].get("UserId") or data[0].get("userId")
        session_id = data[0].get("SessionId") or data[0].get("sessionId")
        features = data[0]
        
        with open("debug_log.txt", "a") as f:
            f.write(f"Received payload: {data}\n")
            f.write(f"Extracted UserID: {user_id}, SessionID: {session_id}\n\n")

    # 🔥 ANALYSIS
    try:
        if confidence >= 0.6:
            analysis = analyze_behavior(
                confidence=confidence,
                features=features
            )
        elif confidence >= 0.3 and confidence < 0.6:
            analysis = {
                "riskLevel": "MEDIUM",
                "reason": "Suspicious pattern detected but with low confidence, monitor behavior"
            }
        else:
                analysis = {
                    "riskLevel": "LOW",
                    "reason": "Behavior appears normal with low anomaly probability , no groq analysis needed  "
                }

    except Exception as e:
        print("GROQ ERROR:", e)
        analysis = {
            "riskLevel": "UNKNOWN",
            "reason": "Analysis service unavailable",
             "context": {
        "page": features.get("CurrentPage") or features.get("currentPage"),
        "stage": features.get("Context") or features.get("context")
        }
    }

    # 🔥 SAVE TO DB (separate block)
    try:
        prediction_label = "Anomaly" if confidence >= 0.5 else "Normal"
        save_prediction(
            user_id=user_id, 
            session_id=session_id, 
            prediction_label=prediction_label,
            confidence_score=confidence,
            risk_level=analysis.get("riskLevel"), 
            analysis_reason=analysis.get("reason")
        )
    except Exception as e:
        print("DB ERROR:", e)


    return {
        # "prediction": "",
        # "label": "",
        "confidence": confidence,
        "analysis": analysis
    }