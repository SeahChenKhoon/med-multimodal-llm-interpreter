import sqlite3
from typing import List

from src.utils.lab_results import LabResult, LabResultList

def read_lab_results_from_sqlite(db_path: str, table_name: str) -> LabResultList:
    """
    Reads lab result records from a SQLite database and returns them as a LabResultList.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table containing lab results.

    Returns:
        LabResultList: A list-like object containing LabResult instances.
                       If the table does not exist or is empty, an empty LabResultList is returned.
    """
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


def export_lab_results_to_sqlite(
    lab_results: List["LabResult"],
    db_path: str,
    table_name: str = "lab_results"
) -> None:
    """
    Export a list of cls_Lab_Result objects to a SQLite database table.

    Args:
        lab_results (List[cls_Lab_Result]): List of results to export.
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to write to.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            filename TEXT,
            test_date TEXT,
            test_common_name TEXT,
            test_name TEXT,
            test_result TEXT,
            test_uom TEXT,
            classification TEXT,
            reason TEXT,
            recommendation TEXT,
            PRIMARY KEY (test_date, test_name)
        )
    """)

    # Insert rows
    for result in lab_results:
        cursor.execute(f"""
            INSERT OR IGNORE INTO {table_name} (
                filename, test_date, test_common_name, test_name, test_result, test_uom, classification, reason, recommendation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.test_filename,
            result.test_date.isoformat() if hasattr(result.test_date, 'isoformat') else result.test_date,
            result.test_common_name,
            result.test_name,
            result.test_result,
            result.test_uom,
            result.classification,
            result.reason,
            result.recommendation 
        ))

    conn.commit()
    conn.close()
