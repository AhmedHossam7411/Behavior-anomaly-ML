from fastapi import APIRouter, Request
from pydantic import BaseModel
from app.save_prediction import save_prediction
from app.groq_service import analyze_behavior

router = APIRouter()


class PredictionRequest(BaseModel):
    data: list


def calibrate_confidence(prob):
    return float(max(0.05, min(0.95, prob)))


def _get(d, *keys, default=0):
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return float(default)


def _rule_based_analysis(confidence: float, features: dict) -> dict:
    """
    Derives a risk level and human-readable reason purely from feature values.
    Used when Groq is unavailable so the demo always shows meaningful output.
    """
    hacking   = _get(features, "HackingStringDetected",  "hackingStringDetected")
    sus_paste = _get(features, "SuspiciousPasteDetected","suspiciousPasteDetected")
    abnormal  = _get(features, "AbnormalInputDetected",  "abnormalInputDetected")
    unauth    = _get(features, "UnauthorizedAttempts",   "unauthorizedAttempts")
    dt_count  = _get(features, "DevToolsShortcutCount",  "devToolsShortcutCount")
    dt_open   = _get(features, "DevToolsDetected",       "devToolsDetected")
    paste     = _get(features, "PasteCount",             "pasteCount")
    std_mouse = _get(features, "StdMouseSpeed",          "stdMouseSpeed", default=999)
    typing    = _get(features, "TypingRate",             "typingRate",    default=999)
    click_r   = _get(features, "ClickRate",              "clickRate",     default=0)
    page      = features.get("CurrentPage") or features.get("currentPage", "unknown")

    signals = []
    risk = "LOW"

    if hacking == 1:
        signals.append("attack string detected in input or URL")
        risk = "HIGH"
    if sus_paste == 1:
        signals.append("suspicious content pasted — bypasses keystroke monitoring")
        risk = "HIGH"
    if abnormal == 1:
        signals.append("input length exceeded 500 characters — possible fuzzing")
        risk = "HIGH"
    if unauth > 2:
        signals.append(f"UnauthorizedAttempts = {int(unauth)} — repeated security challenge bypass attempts")
        risk = "HIGH"
    if dt_open == 1 and dt_count > 3:
        signals.append(f"DevTools open with {int(dt_count)} shortcut keypresses — active session inspection suspected")
        risk = max(risk, "HIGH") if risk == "HIGH" else "HIGH"
    elif dt_open == 1 and dt_count > 0:
        signals.append(f"DevTools detected via viewport dimensions with {int(dt_count)} shortcut(s)")
        if risk == "LOW":
            risk = "MEDIUM"
    if paste > 5:
        signals.append(f"PasteCount = {int(paste)} in this window — unusually high, consistent with scripted input")
        if risk == "LOW":
            risk = "MEDIUM"
    if std_mouse < 0.05 and typing == 0.0 and click_r > 5:
        signals.append(
            f"StdMouseSpeed = {std_mouse:.3f} (near-zero uniformity), TypingRate = 0, "
            f"ClickRate = {click_r:.1f} — robotic automation pattern"
        )
        risk = "HIGH"

    if not signals:
        if confidence >= 0.5:
            signals.append(f"TabPFN anomaly score {confidence:.0%} exceeds normal threshold on page {page}")
            risk = "MEDIUM"
        else:
            signals.append("No significant anomaly signals detected in this window")

    reason = ". ".join(s.capitalize() for s in signals) + "."
    return {"riskLevel": risk, "reason": reason}


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

    # TabPFN verdict
    tabpfn_label   = "Anomaly" if confidence >= 0.5 else "Normal"
    tabpfn_verdict = (
        "Behavioral pattern classified as anomalous by TabPFN"
        if confidence >= 0.5
        else "Behavioral pattern classified as normal by TabPFN"
    )

    # Groq analysis — called whenever TabPFN flags an anomaly (>= 0.5)
    try:
        if confidence >= 0.5:
            analysis = analyze_behavior(confidence=confidence, features=features)
        elif confidence >= 0.25:
            analysis = {
                "riskLevel": "MEDIUM",
                "reason": "Low-confidence anomaly signal — behavior is suspicious but not conclusive. Continue monitoring."
            }
        else:
            analysis = {
                "riskLevel": "LOW",
                "reason": "Behavior is within normal parameters. No anomalous signals detected."
            }
    except Exception as e:
        print("GROQ ERROR:", e)
        analysis = _rule_based_analysis(confidence, features)

    # Save to DB
    try:
        save_prediction(
            user_id=user_id,
            session_id=session_id,
            prediction_label=tabpfn_label,
            confidence_score=confidence,
            risk_level=analysis.get("riskLevel"),
            analysis_reason=analysis.get("reason")
        )
    except Exception as e:
        print("DB ERROR:", e)

    return {
        "confidence": confidence,
        "tabpfn": {
            "score": round(confidence, 4),
            "label": tabpfn_label,
            "verdict": tabpfn_verdict
        },
        "analysis": analysis
    }