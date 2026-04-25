import pyodbc

try:
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\SQLEXPRESS;"
        "DATABASE=GovernmentTaskManagementDB;"
        "Trusted_Connection=yes;"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'BehaviorPredictions'")
    columns = [row[0] for row in cursor.fetchall()]
    print("COLUMNS: ", columns)
except Exception as e:
    print(e)
