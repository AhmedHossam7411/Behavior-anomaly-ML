import pyodbc

def save_prediction(user_id, session_id, prediction_label, confidence_score, risk_level, analysis_reason):
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\SQLEXPRESS;"
        "DATABASE=GovernmentTaskManagementDB;"
        "Trusted_Connection=yes;"
    )

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO BehaviorPredictions (UserId, SessionId, PredictionLabel, ConfidenceScore, RiskLevel, AnalysisReason)
        VALUES (?, ?, ?, ?, ?, ?)
    """, user_id, session_id, prediction_label, confidence_score, risk_level, analysis_reason)

    conn.commit()
    conn.close()
