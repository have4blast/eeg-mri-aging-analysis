# config/load_config.py
import yaml
from pathlib import Path

def load_yaml(path):
    """Load a YAML file and return it as a Python dict."""
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)
