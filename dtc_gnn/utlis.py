from pathlib import Path
from shutil import rmtree
from typing import List


def is_directory_empty(directory_path: Path) -> bool:
    """Check if a directory is empty."""
    return not any(directory_path.glob("*"))


def data_dirs_status(dir_paths: List[Path]) -> bool:
    """
    Checks the status of directories.

    If directories do not exist, creates them.
    If they exist and are not empty, asks the user whether to regenerate data.

    Args:
        dir_paths (List[Path]): A list of Path objects representing directories.

    Returns:
        bool: True if to create or regenerate data, False if left as-is.
    """
    for dir_path in dir_paths:
        if dir_path.exists() and not is_directory_empty(dir_path):
            user_input = ""

            while user_input not in ["yes", "no", "y", "n"]:
                user_input = input(
                    f"The data is already generated. "
                    "Do you want to regenerate its contents? (yes/no): "
                ).strip().lower()

            if user_input.startswith("y"):
                for f_path in dir_path.glob("*"):
                    if f_path.is_file():
                        f_path.unlink()
                    elif f_path.is_dir():
                        rmtree(f_path)
                print(f"Regenerating contents of '{dir_path}'...")
                return True
            print(f"Leaving the contents of '{dir_path}' as-is...")
            return False
        else:
            try:
                dir_path.mkdir(exist_ok=True)
            except OSError as e:
                print(f"Error creating directory '{dir_path}': {e}")
                return False
            return True
