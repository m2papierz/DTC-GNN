import os

from pathlib import Path
from shutil import rmtree
from typing import List


def data_dirs_status(
        dir_paths: List[Path]
) -> bool:
    """
    If directories do not exist, creates them, otherwise, if the user indicates to
    do so cleans them and regenerates the data.
    """
    if any(os.path.exists(path) for path in dir_paths):
        user_input = ""

        while user_input not in ["yes", "no"]:
            user_input = input(
                "The data is already generated.  "
                "Do you want to regenerate it? (yes/no): "
            ).strip().lower()

        if user_input == "yes":
            for dir_ in dir_paths:
                for f_path in dir_.glob("*"):
                    if f_path.is_file():
                        f_path.unlink()
                    elif f_path.is_dir():
                        rmtree(f_path)
            print("Regenerating datasets...")
            return True
        print("Leaving the already existing datasets...")
        return False
    else:
        for dir_ in dir_paths:
            os.mkdir(dir_)
        return True
