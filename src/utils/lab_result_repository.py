
from src.utils.lab_results import LabResult, LabResultList
import sqlite3

def read_lab_results_from_sqlite(db_path: str, table_name: str = "lab_results") -> LabResultList:
    lab_result_list = LabResultList()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the table exists
    cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name=?
    """, (table_name,))
    if cursor.fetchone() is None:
        conn.close()
        return lab_result_list  # Return an empty LabResultList

    # Fetch rows
    cursor.execute(f"""
        SELECT filename, test_date, test_common_name, test_name, test_result, test_uom,
               classification, reason, recommendation
        FROM {table_name}
    """)
    rows = cursor.fetchall()

    lab_result_list.result = [
        LabResult(
            test_filename=row[0],
            test_date=row[1],
            test_common_name=row[2],
            test_name=row[3],
            test_result=row[4],
            test_uom=row[5],
            classification=row[6],
            reason=row[7],
            recommendation=row[8]  
        )
        for row in rows
    ]
    conn.close()
    return lab_result_list