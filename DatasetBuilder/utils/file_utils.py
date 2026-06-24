"""
file_utils.py

Common file utilities used throughout the Dataset Builder.
"""

from pathlib import Path
import json


class FileUtils:

    @staticmethod
    def read_text(path: Path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    @staticmethod
    def write_text(path: Path, text: str):

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(text)

    @staticmethod
    def append_text(path: Path, text: str):

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            path,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(text)

    @staticmethod
    def read_json(path: Path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    @staticmethod
    def get_scenario_folders(dataset_root: Path):

        return sorted(

            [

                folder

                for folder in dataset_root.iterdir()

                if folder.is_dir()
                and folder.name.lower().startswith("scenario_")

            ]

        )