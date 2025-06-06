from typing import List, Tuple, Optional
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

prompt = """
    You are a medical data analyst.

    Your task:
    1. Group lab test names that are functionally or clinically equivalent.
    2. Assign a single standardized name to each group (e.g., "Glucose, Fasting").
    3. For each standardized name, list all test name variants.
    4. Return the result as a table with the following columns:
    - Standard Name
    - Variant Names (semicolon-separated)
    - Notes (optional)

    Test names:
    {joined_tests}
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


def main() -> None:
    """
    Process PDF lab reports in a given folder using an LLM to extract structured lab results.

    Steps:
    - Load all PDF files from the target folder.
    - Extract text content from each PDF.
    - Format the content into an LLM prompt and retrieve structured JSON response.
    - Parse and accumulate lab results into a list of cls_Lab_Result objects.
    - Print each test name with its parsed datetime.

    Raises:
        Any unhandled exceptions during file reading, LLM processing, or JSON parsing.
    """
    data_storage_folder = "data\\selected_data"


    settings_dict = build_settings_dict()

    lab_results: List[cls_Lab_Result] = []

    data_files = _get_data_files(data_storage_folder)
    
    for data_file in data_files:
        print(f"Processing file: {data_file}")
        pdf_content = extract_pdf_text(data_file)
        cleaned_text = clean_pdf_text(pdf_content)

        test_datetime = extract_test_datetime(cleaned_text)
        test_datetime = get_datetime_object(test_datetime)
        prompt = extract_lab_tests_prompt_template.format(lab_result=cleaned_text.strip())
        llm_client = cls_LLM.build_llm_client(settings_dict, prompt)
        response = llm_client.completion()
        if not response.strip():
            print(f"Empty response for {data_file.name}")
            continue

        try:
            data = json.loads(response)

            lab_results.extend(cls_Lab_Result.from_dict(item, data_file.name,test_datetime=test_datetime) 
                               for item in data)

        except json.JSONDecodeError as e:
            print(f"JSON decode error in {data_file.name}: {e}\nResponse: {response}")
            continue
    # lab_test_names = list({result.test_name.strip() for result in lab_results})
    # response = llm.create_client(prompt)
    export_lab_results_to_csv(lab_results, "data\\output\\lab_results_output.csv")


if __name__ == "__main__":
    main()
