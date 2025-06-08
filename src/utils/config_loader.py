from typing import Dict, Any
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