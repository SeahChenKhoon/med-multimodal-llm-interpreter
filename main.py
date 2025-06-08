from typing import List, Tuple, Optional, Dict
from pathlib import Path
from pypdf import PdfReader
import json
import csv
import re
import sqlite3
from src.utils.settings import SETTINGS 
from src.utils.llm_client import LLMClient
import src.utils.config_loader as config_loader
import src.utils.lab_result_repository as sqlite3_lib

from datetime import date, datetime
from src.utils.lab_results import LabResult, LabResultList



def _get_data_files(directory: str) -> List[Path]:
    """
    Recursively retrieve all PDF files from the specified directory.

    Args:
        directory (str): Path to the root directory to search for PDF files.

    Returns:
        List[Path]: A list of Path objects representing all found PDF files.
    """
    return list(Path(directory).rglob("*.pdf"))


def _extract_pdf_text(pdf_file_path: str) -> str:
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

def export_lab_results_to_csv(lab_results: LabResultList, output_path: str) -> None:
    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(["filename", "datetime", "test_common_name", "test_name", "test_result", "test_uom", "classification", "reason", "recommendation"])

        # Rows
        for result in lab_results.result:
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

def _extract_test_datetime(text: str) -> Optional[str]:
    """Extracts the test datetime from PDF content like 'Date: 01 Oct 2022, 08:46 AM'."""
    match = re.search(r"Date:\s*(\d{2} \w{3} \d{4}, \d{2}:\d{2} [AP]M)", text)
    return match.group(1) if match else None

def _clean_pdf_text(text: str) -> str:
    """Removes known footer lines from extracted PDF text."""
    lines = text.splitlines()
    return "\n".join(
        line for line in lines
        if not re.search(r"Generated on:\s*\d{2} \w{3} \d{4}", line)
    )

def _get_date_object(test_datetime) -> Optional[date]:
    """Converts the test_datetime string into a datetime object if possible."""
    try:
        return datetime.strptime(test_datetime, "%d %b %Y, %I:%M %p").date()
    
    except (TypeError, ValueError):
        return None

def get_joined_test_names(lab_results: List[LabResult]) -> str:
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


def _extract_lab_results_from_pdf(data_file, settings_dict, prompt) -> Tuple[List[Dict], "date"]:
    # Extract and clean text from PDF
    pdf_content = _extract_pdf_text(data_file)
    cleaned_text = _clean_pdf_text(pdf_content)

    # Extract and convert test datetime
    test_datetime_str = _extract_test_datetime(cleaned_text)
    test_datetime = _get_date_object(test_datetime_str)
    response = LLMClient.run_prompt(settings_dict, prompt,
                                  {"lab_result":cleaned_text.strip()})

    # Parse LLM response
    try:
        lab_result_dicts = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    return lab_result_dicts, test_datetime

def standardize_test_names(lab_results: List[LabResult], standardization_map):
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
    lab_results: List["LabResult"],
    prompt_template: str, 
    settings_dict: dict
) -> List["LabResult"]:
    # Prepare joined test names and prompt
    predefined_standardizations = format_standardization_map(standardization_map)
    joined_tests = get_joined_test_names(lab_results)


    response = LLMClient.run_prompt(settings_dict=settings_dict,
                                    prompt_template=prompt_template, 
                                    prompt_context={"predefined_standardizations" : predefined_standardizations})

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



def _classify_and_parse_lab_results(
    lab_result_dicts: List[dict],
    settings_dict: dict,
    prompt_template: str,
    data_file_name: str,
    test_date: str
) -> LabResultList:
    """
    Classifies lab results using LLM and parses them into a LabResultList.

    Args:
        lab_result_dicts: Raw extracted lab results from LLM or PDF.
        settings_dict: Configuration dictionary for LLMClient.
        prompt_template: Prompt string with {lab_tests_json} placeholder.
        data_file_name: Name of the data file (for traceability).
        test_date: Date of the test.

    Returns:
        LabResultList: Parsed and classified lab results.
    """
    # Step 1: Run LLM classification
    classified_json = LLMClient.run_prompt(
        settings_dict=settings_dict,
        prompt_template=prompt_template,
        prompt_context={"lab_tests_json": lab_result_dicts}
    )
    # Step 2: Convert JSON string to Python object
    data = json.loads(classified_json)
    # Step 3: Convert to LabResultList
    lab_results = LabResultList()
    lab_results.result = [
        LabResult.from_dict(item, data_file_name, test_date=test_date)
        for item in data
    ]
    return lab_results


def main() -> None:
    # Initialisation
    settings_dict = build_settings_dict()
    config = config_loader.load_config(settings_dict["config_file_path"])    
    data_file = Path(config["path"]["data_file"])
    sqllite_file = Path(config["sqllite"]["file"])
    table_name = config["sqllite"]["table_name"]
    lab_results = LabResultList()
    new_lab_results = LabResultList()
    all_lab_results= LabResultList()
    lab_result_classification_prompt=config["prompt"]["lab_result_classification_prompt"]

    # Read previous lab result from SQLLite
    lab_results = new_lab_results = sqlite3_lib.read_lab_results_from_sqlite(
        sqllite_file, table_name)
    unique_name_pairs =lab_results.get_unique_test_names_str()
    print(f"Total unique_pairs: {unique_name_pairs}")

    # Extract lab result from PDF
    print(f"Processing file: {data_file.name}")
    lab_result_dicts, test_date = _extract_lab_results_from_pdf(
        data_file, settings_dict,
        config["prompt"]["extract_and_classify_lab_tests_prompt_template"])
    
    # Derive Investigation from LLM
    lab_results= _classify_and_parse_lab_results(lab_result_dicts=lab_result_dicts, 
                                    settings_dict=settings_dict,
                                    prompt_template=lab_result_classification_prompt, 
                                    data_file_name=data_file.name, test_date=test_date)
    
    # Update Standard Name
    unmapped_varied_name = lab_results.get_unmapped_test_names_str()
    classified_json = LLMClient.run_prompt(
        settings_dict=settings_dict,
        prompt_template=config["prompt"]["lab_test_name_grouping_prompt_template"],
        prompt_context={"standard_mappings": unique_name_pairs,
                        "new_variants": unmapped_varied_name}
    )
    classified_data = json.loads(classified_json) 
    correction_dict = {
        item["variant_name"]: item["standard_name"]
        for item in classified_data
    }
    lab_results.apply_standardization(correction_dict)

    # Update to  sqllite
    new_lab_results.extend(lab_results)
    export_lab_results_to_sqlite(new_lab_results.result, sqllite_file, table_name)

    # Retrive and output table rows
    lab_results = sqlite3_lib.read_lab_results_from_sqlite(sqllite_file, table_name)
    print(lab_results.describe())


if __name__ == "__main__":
    main()
