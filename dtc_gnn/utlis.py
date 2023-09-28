import os

from pathlib import Path
from shutil import rmtree


def init_directory(dir_: Path):
    """

    """
    if os.path.exists(dir_):
        for f_path in dir_.glob("*"):
            if f_path.is_file():
                f_path.unlink()
            elif f_path.is_dir():
                rmtree(f_path)
    else:
        os.mkdir(dir_)
