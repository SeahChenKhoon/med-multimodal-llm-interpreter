from typing import Dict, Any
from datetime import datetime
import yaml

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        config_path (str): The file path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed YAML configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def format_test_date(test_date):
    if isinstance(test_date, datetime):
        return test_date.strftime("%d/%m/%Y")
    try:
        return datetime.fromisoformat(test_date).strftime("%d/%m/%Y")
    except Exception:
        return test_date or "" 