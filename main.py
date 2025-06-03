from typing import List, Tuple
from pathlib import Path
from pypdf import PdfReader

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
        text_content = extract_pdf_text(data_file)
        print(text_content)


if __name__ == "__main__":
    main()
