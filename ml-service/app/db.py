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
        Context
    FROM BehaviorWindows
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df