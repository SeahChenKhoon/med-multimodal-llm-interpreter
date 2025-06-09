import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
import json
import re
import os
import uuid
from datetime import date, datetime
from pathlib import Path
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple

from pypdf import PdfReader

import src.utils.config_loader as config_loader
from src.utils.lab_results import LabResult, LabResultList
from src.utils.llm_client import LLMClient
from src.utils.settings import SETTINGS

def set_dataframe_column_styles():
    st.markdown("""
    <style>
    /* Date column */
    .dataframe td:nth-child(1),
    .dataframe th:nth-child(1) {
        max-width: 30px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Test Name */
    .dataframe td:nth-child(2),
    .dataframe th:nth-child(2) {
        max-width: 80px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Test Result */
    .dataframe td:nth-child(3),
    .dataframe th:nth-child(3) {
        max-width: 30px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Test Classification */
    .dataframe td:nth-child(4),
    .dataframe th:nth-child(4) {
        max-width: 30px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

                
    /* Recommendation */
    .dataframe td:nth-child(5),
    .dataframe th:nth-child(5) {
        max-width: 300px;
        white-space: normal;
        word-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)

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


def _build_settings_dict() -> dict:
    """
    Builds a dictionary of configuration settings used to initialize LLM clients and other components.

    This function gathers values from a global SETTINGS object, which typically contains environment-specific
    variables such as API keys, endpoints, model identifiers, and file paths.

    Returns:
        dict: A dictionary containing keys for LLM provider settings, API keys, configuration paths, and model info.
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


def _extract_test_datetime(text: str) -> Optional[str]:
    """
    Extracts the test datetime from PDF content.

    The function looks for a line in the format:
        'Date: 01 Oct 2022, 08:46 AM'

    Args:
        text (str): The extracted text content from a PDF file.

    Returns:
        Optional[str]: A string containing the datetime in the format 'DD MMM YYYY, HH:MM AM/PM',
                       or None if no match is found.
    """
    match = re.search(r"Date:\s*(\d{2} \w{3} \d{4}, \d{2}:\d{2} [AP]M)", text)
    return match.group(1) if match else None


def _clean_pdf_text(text: str) -> str:
    """
    Cleans extracted PDF text by removing known footer lines such as those containing
    'Generated on: DD MMM YYYY'.

    Args:
        text (str): Raw text extracted from a PDF.

    Returns:
        str: Cleaned text with footer lines removed.
    """
    lines = text.splitlines()
    return "\n".join(
        line for line in lines
        if not re.search(r"Generated on:\s*\d{2} \w{3} \d{4}", line)
    )


def _get_date_object(test_datetime: Optional[str]) -> Optional[date]:
    """
    Converts a datetime string into a `date` object.

    Args:
        test_datetime (Optional[str]): A string representing the datetime, expected in the format
                                       "%d %b %Y, %I:%M %p" (e.g., "11 Jan 2025, 08:04 AM").

    Returns:
        Optional[date]: A `date` object if conversion succeeds, otherwise `None`.
    """
    try:
        return datetime.strptime(test_datetime, "%d %b %Y, %I:%M %p").date()
    except (TypeError, ValueError):
        return None


def _extract_lab_results_from_pdf(
    data_file, settings_dict: dict, prompt: str
) -> Tuple[List[Dict], date]:
    """
    Extracts lab test results from a PDF file using an LLM and returns the structured results
    along with the date the test was conducted.

    Args:
        data_file: Path or file-like object representing the PDF file.
        settings_dict (dict): Dictionary containing LLM configuration settings.
        prompt (str): Prompt template to guide the LLM extraction.

    Returns:
        Tuple[List[Dict], date]: A tuple containing:
            - A list of dictionaries representing individual lab test results.
            - A date object representing the test date extracted from the PDF.
    
    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON.
    """
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


def save_uploaded_file(uploaded_file, save_dir: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(uploaded_file.name).stem
    extension = Path(uploaded_file.name).suffix
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]
    new_filename = f"{original_name}_{unique_id}{extension}"

    file_path = save_dir / new_filename
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def main() -> None:
    """
    Main orchestration function for processing medical lab results.

    Workflow:
    1. Loads configuration and settings.
    2. Reads previously stored lab results from SQLite.
    3. Extracts new lab results from a specified PDF file using an LLM.
    4. Classifies and parses the extracted lab results using another LLM prompt.
    5. Standardizes test names using previously seen names and the LLM.
    6. Updates the SQLite database with the newly processed lab results.
    7. Reloads the full dataset and prints a summary.

    Returns:
        None
    """
    # Initialisation
    st.set_page_config(page_title="Medical Multimodal LLM Interpreter", layout="wide")
    settings_dict = _build_settings_dict()
    config = config_loader.load_config(settings_dict["config_file_path"])    
    data_file = Path(config["path"]["data_file"])
    processing_dir = Path(config["path"]["processing_dir"])
    sqllite_file = Path(config["sqllite"]["file"])
    table_name = config["sqllite"]["table_name"]
    lab_results = LabResultList()
    lab_result_classification_prompt=config["prompt"]["lab_result_classification_prompt"]

    # Read previous lab result from SQLLite
    db_lab_results = LabResultList.read_lab_results_from_sqlite(
        sqllite_file, table_name)
    unique_name_pairs =db_lab_results.get_unique_test_names_str()
    
    st.title("🩺 Medical Multimodal LLM Interpreter")
    uploaded_files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("🔄 Process Uploaded Files"):
        
            for uploaded_file in uploaded_files:
                full_path_str = str(processing_dir / uploaded_file.name)
                data_file = save_uploaded_file(uploaded_file, processing_dir)

                # Extract lab result from PDF
                st.write(f"Processing file: {uploaded_file.name}")
                lab_result_dicts, test_date = _extract_lab_results_from_pdf(
                    data_file, settings_dict,
                    config["prompt"]["extract_and_classify_lab_tests_prompt_template"])
                
                # Derive Investigation from LLM
                lab_results= _classify_and_parse_lab_results(lab_result_dicts=lab_result_dicts, 
                                                settings_dict=settings_dict,
                                                prompt_template=lab_result_classification_prompt, 
                                                data_file_name=full_path_str, test_date=test_date,
                                                )
                
                # Update Standard Name
                lab_results.standardize_test_names(
                        settings_dict=settings_dict, 
                        prompt_template=config["prompt"]["lab_test_name_grouping_prompt_template"],
                        unique_name_pairs=unique_name_pairs)

                # Output result 
                lab_results.export_lab_results_to_sqlite(sqllite_file, table_name)

                if data_file.exists():
                    data_file.unlink()

            # Retrive and output table rows
            lab_results = LabResultList.read_lab_results_from_sqlite(sqllite_file, table_name)
            lab_results.export_to_csv(config["path"]["csv_file"])

            df = lab_results.lab_results_to_dataframe()
            
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_column("test_date", width=80)
            gb.configure_column("test_name", width=200)
            gb.configure_column("test_result", width=50)
            gb.configure_column("classification", width=70)
            gb.configure_column("recommendation", width=1000, wrapText=True, autoHeight=True)
            grid_options = gb.build()

            AgGrid(df, gridOptions=grid_options, fit_columns_on_grid_load=True)

            # Filter rows with non-empty recommendations
            recommended_df = df[df["recommendation"].notna() & (df["recommendation"].str.strip() != "")]

            # Extract test names
            test_name_list = recommended_df["test_name"].unique().tolist()
            # Display
            st.subheader("🩺 Tests with Recommendations")
            
            for test_name in test_name_list:
                st.markdown(f"### 🧪 Test: {test_name}")

                # Step 3: Filter original df for all records of this test_name
                filtered_df = df[df["test_name"] == test_name].sort_values(by="test_date", ascending=False)
                filtered_df["test_date"] = pd.to_datetime(filtered_df["test_date"], dayfirst=True, errors="coerce")
                filtered_df = filtered_df.sort_values(by="test_date", ascending=False)
                filtered_df["test_date"] = filtered_df["test_date"].dt.strftime("%d/%m/%Y")

                # Step 4: Configure AgGrid
                gb = GridOptionsBuilder.from_dataframe(filtered_df)
                gb.configure_column("test_date", width=80)
                gb.configure_column("test_name", width=200)
                gb.configure_column("test_result", width=100)
                gb.configure_column("classification", width=100)
                gb.configure_column("recommendation", width=800, wrapText=True, autoHeight=True)
                grid_options = gb.build()

                # Step 5: Display the data
                AgGrid(filtered_df, gridOptions=grid_options, fit_columns_on_grid_load=True)

if __name__ == "__main__":
    main()
