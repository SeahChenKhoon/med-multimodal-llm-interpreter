from typing import List, Tuple, Optional, Dict
from pathlib import Path
from pypdf import PdfReader
import json
import csv
import re
from src.utils.settings import SETTINGS 
import src.utils.cls_LLM as cls_LLM
from datetime import datetime
from src.utils.cls_Lab_Result import cls_Lab_Result

extract_lab_tests_prompt_template = """
You are an information extraction assistant. Extract the following fields from a medical lab test document:

- datetime: The exact date and time the test was performed or reported.
- test_name: The name of the medical test performed (e.g., "ALBUMIN, U (NHGD)").
- test_result: The measured value (e.g., "24.0").
- test_uom: The measured unit (e.g., mg/L)
- ref_range: The reference range if provided. It is usually a numeric range like "10-50", "0.0-2.5", or ">=5.0". Do NOT interpret it as a date.
- diagnostic: Any medical interpretation or comment, if available.

Here is the text to process:
{lab_result}

Return the results as a list of JSON objects, one per test, in this format:
[
    {{
    "datetime": "...",
    "test_name": "...",
    "test_result": "...",
    "test_uom": "...",
    "ref_range": "...",
    "diagnostic": "..."
    }},
    ...
]
No Markdown formatting, explanations, or docstrings. Do NOT wrap your output in backticks.
"""

lab_test_name_grouping_prompt_template  = """
    You are a medical data analyst.

    Your task:
    1. Group lab test names that are functionally or clinically equivalent.
    2. Assign a single standardized name to each group (e.g., "Glucose, Fasting").
    3. For each standardized name, list all test name variants.
    4. Return the result as a table with the following columns:
    - Standard Name
    - Variant Names (semicolon-separated)

    Test names:
    {joined_tests}

    No Markdown formatting, explanations, or docstrings. Do NOT wrap your output in backticks.
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
        "llm_temperature": SETTINGS.LLM_TEMPERATURE,
        "llm_model": SETTINGS.LLM_MODEL
    }


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
        writer.writerow(["filename", "datetime", "test_name", "test_result", "test_uom", "ref_range", "diagnostic"])

        # Rows
        for result in lab_results:
            writer.writerow([
                result.test_filename,
                result.test_datetime,
                result.test_name,
                result.test_result,
                result.test_uom,
                f"'{result.ref_range}" if result.ref_range else "",
                result.diagnostic
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

def get_datetime_object(test_datetime) -> Optional[datetime]:
    """Converts the test_datetime string into a datetime object if possible."""
    try:
        return datetime.strptime(test_datetime, "%d %b %Y, %I:%M %p")
    except (TypeError, ValueError):
        return None

def get_joined_test_names(lab_results: List[cls_Lab_Result]) -> str:
    test_names = sorted({result.test_name.strip() for result in lab_results})
    return "\n".join(test_names)

def extract_lab_results_from_pdf(data_file, settings_dict) -> Tuple[List[Dict], "datetime"]:
    """
    Extracts structured lab result data and test datetime from a PDF using an LLM.

    Args:
        data_file: A file-like object representing the PDF file.
        settings_dict: Dictionary containing LLM client configuration.

    Returns:
        Tuple[List[Dict], datetime]: A list of extracted lab result dictionaries and the test datetime.
    """
    # Extract and clean text from PDF
    pdf_content = extract_pdf_text(data_file)
    cleaned_text = clean_pdf_text(pdf_content)

    # Extract and convert test datetime
    test_datetime_str = extract_test_datetime(cleaned_text)
    test_datetime = get_datetime_object(test_datetime_str)

    # Prepare and send prompt to LLM
    prompt = extract_lab_tests_prompt_template.format(lab_result=cleaned_text.strip())
    llm_client = cls_LLM.build_llm_client(settings_dict, prompt)
    response = llm_client.completion()

    # Parse LLM response
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    return data, test_datetime

def extract_lab_results_from_pdf(data_file, settings_dict) -> Tuple[List[Dict], "datetime"]:
    """
    Extracts structured lab result data and test datetime from a PDF using an LLM.

    Args:
        data_file: A file-like object representing the PDF file.
        settings_dict: Dictionary containing LLM client configuration.

    Returns:
        Tuple[List[Dict], datetime]: A list of extracted lab result dictionaries and the test datetime.
    """
    # Extract and clean text from PDF
    pdf_content = extract_pdf_text(data_file)
    cleaned_text = clean_pdf_text(pdf_content)

    # Extract and convert test datetime
    test_datetime_str = extract_test_datetime(cleaned_text)
    test_datetime = get_datetime_object(test_datetime_str)

    # Prepare and send prompt to LLM
    prompt = extract_lab_tests_prompt_template.format(lab_result=cleaned_text.strip())
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


def standardize_lab_result_names_with_llm(
    lab_results: List["cls_Lab_Result"],
    settings_dict: dict
) -> List["cls_Lab_Result"]:
    """
    Uses an LLM to generate a standardization map for lab test names and applies it
    to the provided list of cls_Lab_Result objects.

    Args:
        lab_results: List of cls_Lab_Result objects.
        settings_dict: Dictionary with LLM configuration.

    Returns:
        Updated list of cls_Lab_Result with standardized test_name fields.
    """
    # Prepare joined test names and prompt
    joined_tests = get_joined_test_names(lab_results)
    prompt = lab_test_name_grouping_prompt_template.format(joined_tests=joined_tests)

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
    return standardize_test_names(lab_results, standardization_map)

def main() -> None:
    data_storage_folder = "data\\all_data"


    settings_dict = build_settings_dict()

    lab_results: List[cls_Lab_Result] = []

    data_files = _get_data_files(data_storage_folder)
    
    for data_file in data_files:
        print(f"Processing file: {data_file}")
        data, test_datetime = extract_lab_results_from_pdf(data_file, settings_dict)
        lab_results.extend(cls_Lab_Result.from_dict(item, data_file.name,test_datetime=test_datetime) 
                            for item in data)

    lab_results = standardize_lab_result_names_with_llm(lab_results, settings_dict)
    
    export_lab_results_to_csv(lab_results, "data\\output\\lab_results_output.csv")


if __name__ == "__main__":
    main()
