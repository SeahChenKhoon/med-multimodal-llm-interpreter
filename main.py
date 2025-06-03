from typing import List, Tuple
from pathlib import Path
from pypdf import PdfReader


llm_prompt_template = """
You are an information extraction assistant. Extract the following fields from a medical lab test document:

- datetime: The exact date and time the test was performed or reported.
- test_name: The name of the medical test performed.
- test_result: The measured value and unit of the test.
- ref_range: The reference range if provided (e.g., normal limits).
- diagnostic: Any medical interpretation or comment (if available).

Here is the text to process:
{text}

Return the result strictly in this JSON format:
{{
  "datetime": "...",
  "test_name": "...",
  "test_result": "...",
  "ref_range": "...",
  "diagnostic": "..."
}}
"""

def _get_data_files(directory: str) -> Tuple[List[Path], int, List[dict]]:
    return list(Path(directory).rglob("*.pdf"))

def extract_pdf_text(pdf_file_path: str) -> str:
    reader = PdfReader(pdf_file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def main() -> None:
    data_storage_folder="temp_data"

    data_files = _get_data_files(data_storage_folder)
    for data_file in data_files:
        pdf_content = extract_pdf_text(data_file)
        prompt = llm_prompt_template.format(text=pdf_content.strip())
        print(prompt)

if __name__ == "__main__":
    main()
