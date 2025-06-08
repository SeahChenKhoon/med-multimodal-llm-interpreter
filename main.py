from typing import List, Tuple, Optional, Dict
from pathlib import Path
from pypdf import PdfReader
import json
import csv
import re
import sqlite3
from src.utils.settings import SETTINGS 
import src.utils.cls_LLM as cls_LLM
from datetime import date, datetime
from src.utils.cls_Lab_Result import cls_Lab_Result

extract_and_classify_lab_tests_prompt_template = """
You are a medical assistant AI. Given a medical lab test document, extract the following fields from the text:

- datetime: The exact date and time the test was performed or reported.
- test_name: The name of the medical test performed (e.g., "ALBUMIN, U (NHGD)").
- test_result: The measured value (e.g., "24.0").
- test_uom: The measured unit (e.g., mg/L).

**Input text:**
{lab_result}

**Expected Output:**
A list of JSON objects. One per test. Each object must follow this structure:
[
    {{
        "datetime": "...",
        "test_name": "...",
        "test_result": "...",
        "test_uom": "..."
    }},
    ...
]

Do NOT include Markdown formatting, explanations, or wrap the output in backticks.
"""

lab_test_name_grouping_prompt_template = """
You are a medical data analyst.

Your task:
1. Group lab test names that are functionally or clinically equivalent.
2. Assign a single standardized name to each group (e.g., "Glucose, Fasting").
3. For each standardized name:
   - Extract and list test name variants that appear as field labels, result headers, or measurement titles (e.g., "HbA1c (%)").
   - Prefer these over descriptive names from narrative sections (e.g., avoid using "Haemoglobin A1c (HbA1c)" if "HbA1c (%)" appears as a result header).
   - Additionally, include a separate row where the Standard Name maps to itself (e.g., "Albumin, Urine" -> "Albumin, Urine").
4. Use the predefined standardization examples below when relevant.
5. Return the result as a table with the following columns:
   - Standard Name
   - Variant Names (semicolon-separated)

Predefined standardizations:
{predefined_standardizations}

Test names to group:
{joined_tests}

No Markdown formatting, explanations, or docstrings. Do NOT wrap your output in backticks.
"""

lab_result_classification_prompt = """
You are a medical assistant AI. Interpret the following lab test results and classify whether each test is normal, high, or low.

For each test, provide:
1. "classification": "normal", "high", or "low"
2. "reason": Brief explanation for the classification. If no reference range is given, use medically accepted typical ranges or clinical judgment.
3. "recommendation": Optional; include only if the result is abnormal.

Here are the test results to interpret:

{lab_tests_json}

Respond in the following format, one object per test:
[
    {{
        "datetime": "...",
        "test_name": "...",
        "test_result": "...",
        "test_uom": "...",
        "classification": "...",
        "reason": "...",
        "recommendation": "..."  // blank if normal
    }},
    ...
]

Do NOT include explanations outside the JSON. Do NOT use Markdown or wrap the output in backticks.
"""
def _get_data_files(directory: str) -> List[Path]:
    """
    Recursively retrieve all PDF files from the specified directory.

    Args:
        directory (str): Path to the root directory to search for PDF files.

    Returns:
        List[Path]: A list of Path objects representing all found PDF files.
    """
    return list(Path(directory).rglob("*.pdf"))


def extract_pdf_text(pdf_file_path: str) -> str:
    """
    Extract and return text content from all pages of a PDF file.

    Args:
        pdf_file_path (str): Path to the PDF file.

    Returns:
        str: Combined text extracted from each page, separated by newlines.
    """
    reader = PdfReader(pdf_file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def build_settings_dict() -> dict:
    """
    Build and return the LLM settings dictionary from global SETTINGS.

    Returns:
        dict: Dictionary containing LLM-related configuration values.
    """
    return {
        "provider": SETTINGS.LLM_PROVIDER,
        "openai_api_key": SETTINGS.OPENAI_API_KEY,
        "config_dir": SETTINGS.CONFIG_DIR,
        "azure_openai_api_key": SETTINGS.AZURE_OPENAI_API_KEY,
        "azure_openai_endpoint": SETTINGS.AZURE_OPENAI_ENDPOINT,
        "azure_openai_deployment": SETTINGS.AZURE_OPENAI_DEPLOYMENT,
        "azure_api_version": SETTINGS.AZURE_API_VERSION,
        "llm_model": SETTINGS.LLM_MODEL,
        "config_file_path": SETTINGS.CONFIG_FILE_PATH
    }

def export_lab_results_to_sqlite(
    lab_results: List["cls_Lab_Result"],
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

    cursor.execute(f"""
        DROP TABLE IF EXISTS  {table_name}""")

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

def export_lab_results_to_csv(lab_results: List["cls_Lab_Result"], output_path: str) -> None:
    """
    Export a list of cls_Lab_Result objects to a CSV file.

    Args:
        lab_results (List[cls_Lab_Result]): List of results to export.
        output_path (str): Path to the CSV file to write.
    """
    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(["filename", "datetime", "test_common_name", "test_name", "test_result", "test_uom", "classification", "reason", "recommendation"])

        # Rows
        for result in lab_results:
            writer.writerow([
                result.test_filename,
                result.test_date,
                result.test_common_name,
                result.test_name,
                result.test_result,
                result.test_uom,
                result.classification,
                result.reason,
                result.recommendation
            ])

def extract_test_datetime(text: str) -> Optional[str]:
    """Extracts the test datetime from PDF content like 'Date: 01 Oct 2022, 08:46 AM'."""
    match = re.search(r"Date:\s*(\d{2} \w{3} \d{4}, \d{2}:\d{2} [AP]M)", text)
    return match.group(1) if match else None

def clean_pdf_text(text: str) -> str:
    """Removes known footer lines from extracted PDF text."""
    lines = text.splitlines()
    return "\n".join(
        line for line in lines
        if not re.search(r"Generated on:\s*\d{2} \w{3} \d{4}", line)
    )

def get_date_object(test_datetime) -> Optional[date]:
    """Converts the test_datetime string into a datetime object if possible."""
    try:
        return datetime.strptime(test_datetime, "%d %b %Y, %I:%M %p").date()
    
    except (TypeError, ValueError):
        return None

def get_joined_test_names(lab_results: List[cls_Lab_Result]) -> str:
    test_names = sorted({result.test_name.strip() for result in lab_results})
    return "\n".join(test_names)

def load_test_name_mappings_from_sqlite(
    db_path: str,
    table_name: str = "test_name_mappings"
) -> Dict[str, str]:
    """
    Loads test name variant-to-standard mappings from a SQLite table into a dictionary.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table containing the mappings.

    Returns:
        Dict[str, str]: A dictionary where keys are variant names and values are standard names.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the table exists
    cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name=?
    """, (table_name,))
    if cursor.fetchone() is None:
        conn.close()
        return {}

    # Retrieve mappings
    cursor.execute(f"SELECT variant_name, standard_name FROM {table_name}")
    rows = cursor.fetchall()

    conn.close()
    return {variant: standard for variant, standard in rows}


def extract_lab_results_from_pdf(data_file, settings_dict) -> Tuple[List[Dict], "date"]:
    # Extract and clean text from PDF
    pdf_content = extract_pdf_text(data_file)
    cleaned_text = clean_pdf_text(pdf_content)

    # Extract and convert test datetime
    test_datetime_str = extract_test_datetime(cleaned_text)
    test_datetime = get_date_object(test_datetime_str)
    # Prepare and send prompt to LLM
    prompt = extract_and_classify_lab_tests_prompt_template.format(lab_result=cleaned_text.strip())
    llm_client = cls_LLM.build_llm_client(settings_dict, prompt)
    response = llm_client.completion()

    # Parse LLM response
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    return data, test_datetime

def standardize_test_names(lab_results: List[cls_Lab_Result], standardization_map):
    for result in lab_results:
        if result.test_name:
            normalized_name = result.test_name.strip().lower()
            if normalized_name in standardization_map:
                result.test_name = standardization_map[normalized_name]
    return lab_results

def format_standardization_map(standardization_map: Dict[str, str]) -> str:
    output_str = "Variant Name -> Standard Name\n"
    output_str += "-" * 40 + "\n"
    
    for variant, standard in standardization_map.items():
        output_str += f"{variant} -> {standard}\n"
    return output_str


def standardize_lab_result_names_with_llm(
    standardization_map: Dict[str, str],
    lab_results: List["cls_Lab_Result"],
    settings_dict: dict
) -> List["cls_Lab_Result"]:
    # Prepare joined test names and prompt
    predefined_standardizations = format_standardization_map(standardization_map)
    joined_tests = get_joined_test_names(lab_results)
    prompt = lab_test_name_grouping_prompt_template.format(joined_tests=joined_tests,\
                                            predefined_standardizations=predefined_standardizations)
    
    # Send prompt to LLM
    llm_client = cls_LLM.build_llm_client(settings_dict, prompt)
    response = llm_client.completion()

    # Parse LLM table output into a mapping
    standardization_map: Dict[str, str] = {}

    for line in response.strip().split("\n"):
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 2:
            continue
        standard_name, variants = parts
        for variant in variants.split(";"):
            normalized_variant = variant.strip().lower()
            standardization_map[normalized_variant] = standard_name

    # Apply standardization
    return standardize_test_names(lab_results, standardization_map), standardization_map

def extract_data_to_str(data: list) -> str:
    output_str = ""
    for result in data:
        for attr in [
            "test_filename", "test_date", "test_common_name", "test_name", "test_result", "test_uom",
            "classification", "reason", "recommendation"
        ]:
            value = getattr(result, attr, "")
            output_str += f"{attr}: {value}\n"
        output_str += "-" * 20 + "\n"
    return output_str

def save_test_name_mappings_to_sqlite(
    mapping: Dict[str, str],
    db_path: str,
    table_name: str = "test_name_mappings"
) -> None:
    """
    Saves a dictionary of variant-to-standard test name mappings into a SQLite table.

    Args:
        mapping (Dict[str, str]): Dictionary where keys are variant names and values are standard names.
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to store the mappings.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            variant_name TEXT PRIMARY KEY,
            standard_name TEXT
        )
    """)

    # Optional: skip dictionary header if present
    for variant, standard in mapping.items():
        if variant.lower() == "variant names" and standard.lower() == "standard name":
            continue
        cursor.execute(f"""
            INSERT OR REPLACE INTO {table_name} (variant_name, standard_name)
            VALUES (?, ?)
        """, (variant, standard))

    conn.commit()
    conn.close()

def read_lab_results_from_sqlite(db_path: str, table_name: str = "lab_results") -> List[cls_Lab_Result]:
    """
    Reads lab results from a SQLite database and returns them as a list of cls_Lab_Result objects.
    Returns an empty list if the table does not exist.

    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to read from.

    Returns:
        List[cls_Lab_Result]: List of lab result objects or empty list if table not found.
    """
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

    lab_results = [
        cls_Lab_Result(
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
    return lab_results

import src.utils.util as util

def main() -> None:
    data_storage_folder = "data\\all_data"
    data_file = Path(".\\data\\selected_data\\Lab_Test_Results_030625(5).pdf")
    settings_dict = build_settings_dict()
    lab_results: List[cls_Lab_Result] = []
    all_lab_results: List[cls_Lab_Result] = []
    print(f"Processing file: {data_file.name}")
    data_list, test_date = extract_lab_results_from_pdf(data_file, settings_dict)
    

    prompt = lab_result_classification_prompt.format(lab_tests_json=data_list)
    llm_client = cls_LLM.build_llm_client(settings_dict, prompt)
    response = llm_client.completion()
    data = json.loads(response)
    lab_results.extend(cls_Lab_Result.from_dict(item, data_file.name,test_date=test_date) 
                        for item in data)
    all_lab_results.extend(lab_results)
    # standardization_map=load_test_name_mappings_from_sqlite("data\\output\\med_multi_modal.db", "test_name_mappings")
    # lab_results, new_standardization_map = standardize_lab_result_names_with_llm(standardization_map, all_lab_results, settings_dict)
    # if len(new_standardization_map)==0:
    #     new_standardization_map = standardization_map
    # save_test_name_mappings_to_sqlite(new_standardization_map, "data\\output\\med_multi_modal.db", "test_name_mappings")

    export_lab_results_to_csv(all_lab_results, "data\\output\\lab_results_output.csv")
    export_lab_results_to_sqlite(all_lab_results, "data\\output\\med_multi_modal.db", "lab_results")
    lab_results = read_lab_results_from_sqlite("data\\output\\med_multi_modal.db","lab_results")
    print(len(lab_results))
    print(extract_data_to_str(lab_results))

    # standardization_map_str = format_standardization_map(new_standardization_map)
    # print(standardization_map_str)

if __name__ == "__main__":
    main()
