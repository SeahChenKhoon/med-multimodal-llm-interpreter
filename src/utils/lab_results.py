import json
import os
import sqlite3
import pandas as pd
from datetime import date, datetime
from typing import Optional, List, Dict, Set, Tuple

from src.utils.llm_client import LLMClient

class LabResult:
    """
    Initialize a LabResult instance representing a single lab test result.

    Args:
        test_filename (Optional[str]): The name of the file from which the result was extracted.
        test_date (Optional[datetime]): The date and time the test was conducted.
        test_common_name (Optional[str]): A standardized or simplified name for the test.
        test_name (Optional[str]): The raw or detailed name of the test.
        test_result (Optional[float]): The numerical result of the test.
        test_uom (Optional[str]): The unit of measurement used for the test result.
        classification (Optional[str]): Classification such as 'high', 'low', or 'normal'.
        reason (Optional[str]): Explanation or reasoning behind the classification.
        recommendation (Optional[str]): Medical or practical recommendations based on the result.
    """    
    def __init__(
        self,
        test_filename: Optional[str] = None,
        test_date: Optional[datetime] = None,
        test_common_name: Optional[str] = None,
        test_name: Optional[str] = None,
        test_result: Optional[float] = None,
        test_uom: Optional[str] = None,
        classification: Optional[str] = None,
        reason: Optional[str] = None,
        recommendation: Optional[str] = None,
    ):
        self.test_filename = test_filename
        self.test_date = test_date
        self.test_common_name = test_common_name
        self.test_name = test_name
        self.test_result = test_result
        self.test_uom = test_uom
        self.classification = classification
        self.reason = reason
        self.recommendation = recommendation

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Optional[str]],
        filename: Optional[str] = None,
        test_date: Optional[str] = None
    ) -> "LabResult":
        """
        Create a LabResult instance from a dictionary of test data.

        Args:
            data (Dict[str, Optional[str]]): A dictionary containing keys corresponding to lab result fields.
            filename (Optional[str]): The name of the file where the lab result was found.
            test_date (Optional[str]): An optional override for the test date. If not provided, data["test_date"] is used.

        Returns:
            LabResult: An instance of the LabResult class populated with the provided data.
        """
        return cls(
            test_filename= filename,
            test_date= test_date or data.get("test_date"),
            test_common_name=data.get("test_comon_name"),
            test_name=data.get("test_name"),
            test_result=data.get("test_result"),
            test_uom=data.get("test_uom"),
            classification=data.get("classification"),
            reason=data.get("reason"),
            recommendation=data.get("recommendation"),
        )


class LabResultList:
    """
    A container class for holding and manipulating a list of LabResult objects.
    """

    def __init__(self):
        """
        Initializes an empty list to store LabResult instances.
        """
        self.result: List[LabResult] = []


    def get_unique_test_names_str(self) -> str:
        """
        Constructs a newline-separated string of unique (test_common_name, test_name) pairs
        from the lab results.

        Returns:
            str: A string where each line represents a unique pair in the format 
                'test_common_name -> test_name'. If any field is None, it is replaced with an empty string.
        """
        unique_pairs = {
            (result.test_common_name, result.test_name)
            for result in self.result
        }
        return "\n".join(f"{common or ''} -> {name or ''}" for common, name in unique_pairs)
    

    def get_unmapped_test_names_str(self) -> str:
        """
        Returns a newline-separated string of LabResult.test_name values
        where test_common_name is None.

        This is typically used to identify test names that need standardization.

        Returns:
            str: A newline-separated string of test_name values lacking a standard name.
        """
        return "\n".join(
            result.test_name for result in self.result
            if result.test_common_name is None and result.test_name
        ) 
    

    def apply_standardization(self, correction_dict: Dict[str, str]) -> None:
        """
        Applies standardization to each LabResult's `test_common_name` based on a correction dictionary.

        Args:
            correction_dict (Dict[str, str]): A mapping from variant `test_name` to standardized `test_common_name`.
        """
        for result in self.result:
            if result.test_name and result.test_name in correction_dict:
                result.test_common_name = correction_dict[result.test_name]


    def describe(self) -> str:
        """
        Returns a formatted string summarizing all LabResult entries in the list.
        Includes the total count and all attributes of each LabResult instance.

        Returns:
            str: A human-readable summary of the lab results.
        """
        descriptions = [f"Total Lab Results: {len(self.result)}"]
        for idx, result in enumerate(self.result, start=1):
            lines = [f"\nLabResult #{idx}"]
            for attr, value in vars(result).items():
                lines.append(f"  {attr}: {value}")
            descriptions.append("\n".join(lines))
        return "\n".join(descriptions)
    
    
    def standardize_test_names(
        self, settings_dict: dict, prompt_template: str, unique_name_pairs: str
    ) -> None:
        """
        Uses an LLM to standardize test names in the LabResultList by updating test_common_name.

        Args:
            settings_dict (dict): Dictionary of LLM settings and API keys.
            prompt_template (str): Template used to generate the LLM prompt.
            unique_name_pairs (str): String of existing (common_name -> test_name) mappings.
        """
        unmapped_names = self.get_unmapped_test_names_str()
        if not unmapped_names.strip():
            return  # Nothing to standardize

        response = LLMClient.run_prompt(
            settings_dict=settings_dict,
            prompt_template=prompt_template,
            prompt_context={
                "standard_mappings": unique_name_pairs,
                "new_variants": unmapped_names
            }
        )

        try:
            classified_data = json.loads(response)
            correction_dict = {
                item["variant_name"]: item["standard_name"]
                for item in classified_data
            }
            self.apply_standardization(correction_dict)
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Standardization failed: {e}")


    def export_to_csv(self, output_path: str) -> None:
        """
        Appends lab results to a CSV file.

        Args:
            output_path (str): Path to the CSV file to append or create.
        """
        rows = [
            {
                "filename": result.test_filename,
                "test_date": result.test_date,
                "test_common_name": result.test_common_name,
                "test_name": result.test_name,
                "test_result": result.test_result,
                "test_uom": result.test_uom,
                "classification": result.classification,
                "reason": result.reason,
                "recommendation": result.recommendation
            }
            for result in self.result
        ]
        df = pd.DataFrame(rows)
        file_exists = os.path.exists(output_path)
        df.to_csv(output_path, mode='a', index=False, header=not file_exists)


    def read_lab_results_from_sqlite(db_path: str, table_name: str) -> "LabResultList":
        lab_result_list = LabResultList()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name=?
        """, (table_name,))
        if cursor.fetchone() is None:
            conn.close()
            return []

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
        self,
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
        for result in self.result:
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
