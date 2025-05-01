# src/utils/paths.py
from pathlib import Path

# This file's path: /path/to/your_project_folder/src/utils/paths.py
# We want the parent of the parent of this file's directory
# Path(__file__) -> .../src/utils/paths.py
# Path(__file__).parent -> .../src/utils
# Path(__file__).parent.parent -> .../src
# Path(__file__).parent.parent.parent -> /path/to/your_project_folder

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# You can also define other key paths relative to the root
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
TEST_DIR = PROJECT_ROOT / "test"

# Example usage:
# MY_DATA_FILE = DATA_DIR / "my_data.csv"
