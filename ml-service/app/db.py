import pyodbc
import pandas as pd

def get_data():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\SQLEXPRESS;"
        "DATABASE=GovernmentTaskManagementDB;"
        "Trusted_Connection=yes;"
    )

    query = """
    SELECT 
        AvgMouseSpeed,
        StdMouseSpeed,
        MouseMoveCount,
        AvgMouseIdle,
        AvgClickDuration,
        ClickCount,
        AvgClickInterval,
        AvgDwell,
        AvgFlight,
        KeyEventCount,
        TypingRate,
        ClickRate,
        MouseMoveRate,
        currentPage,
        Context,
        HackingStringDetected,
        DetectedPatterns,
        PasteCount,
        SuspiciousPasteDetected,
        DevToolsShortcutCount,
        AbnormalInputDetected,
        DevToolsDetected,
        UnauthorizedAttempts
    FROM BehaviorWindows
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Fill NULLs with 0 — older rows may lack newer columns added via migrations
    df = df.fillna(0)

    return df