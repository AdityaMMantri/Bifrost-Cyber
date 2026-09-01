"""
file_utils.py

Shared filesystem utilities used throughout SOGARL.

This module ONLY handles generic filesystem operations.

It does NOT:

    - understand scenarios
    - parse scenario.md
    - load metadata
    - select relevant/noise files
    - build prompts
    - generate model responses
    - calculate Oracle rewards
    - perform GRPO

The purpose of this file is to avoid repeating low-level
file and directory operations across the project.

Typical users:

    scenario_loader.py
        -> read_text()

    metadata_loader.py
        -> read_json()

    code_loader.py
        -> read_text()

    checkpoint.py
        -> ensure_directory()
        -> write_json()

    replay_buffer.py
        -> append_jsonl()

    metrics_logger.py
        -> write_json()
        -> write_text()
"""


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional


class FileUtils:
    """
    Generic filesystem helper functions.

    All methods are static because this class does not maintain
    filesystem state.
    """

    # ========================================================================
    # PATH HELPERS
    # ========================================================================

    @staticmethod
    def to_path(
        path: str | Path,
    ) -> Path:
        """
        Convert a string or Path into a Path object.

        Does not require the path to exist.
        """

        return Path(path).expanduser()

    # ========================================================================
    # EXISTENCE CHECKS
    # ========================================================================

    @staticmethod
    def file_exists(
        path: str | Path,
    ) -> bool:
        """
        Return True if the given path exists and is a file.
        """

        return (
            FileUtils.to_path(path).is_file()
        )

    @staticmethod
    def directory_exists(
        path: str | Path,
    ) -> bool:
        """
        Return True if the given path exists and is a directory.
        """

        return (
            FileUtils.to_path(path).is_dir()
        )

    @staticmethod
    def exists(
        path: str | Path,
    ) -> bool:
        """
        Return True if the given path exists.
        """

        return (
            FileUtils.to_path(path).exists()
        )

    # ========================================================================
    # DIRECTORY OPERATIONS
    # ========================================================================

    @staticmethod
    def ensure_directory(
        path: str | Path,
    ) -> Path:
        """
        Create a directory if it does not already exist.

        Parent directories are created automatically.

        Returns:
            Path object representing the directory.
        """

        directory = FileUtils.to_path(
            path
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    # ========================================================================
    # TEXT READING
    # ========================================================================

    @staticmethod
    def read_text(
        path: str | Path,
        encoding: str = "utf-8",
    ) -> str:
        """
        Read a text file and return its contents.

        Used for:

            - scenario.md
            - source-code files
            - logs
            - text configuration files

        No file-type or extension assumptions are made.
        """

        file_path = FileUtils.to_path(
            path
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"File does not exist: "
                f"{file_path}"
            )

        if not file_path.is_file():

            raise IsADirectoryError(
                f"Expected a file but found "
                f"a directory: {file_path}"
            )

        try:

            return file_path.read_text(
                encoding=encoding
            )

        except UnicodeDecodeError as exc:

            raise UnicodeDecodeError(
                exc.encoding,
                exc.object,
                exc.start,
                exc.end,
                (
                    f"Unable to decode file "
                    f"'{file_path}' using "
                    f"encoding '{encoding}'."
                ),
            ) from exc

    # ========================================================================
    # TEXT WRITING
    # ========================================================================

    @staticmethod
    def write_text(
        path: str | Path,
        content: str,
        encoding: str = "utf-8",
    ) -> Path:
        """
        Write text content to a file.

        Parent directories are created automatically.
        """

        file_path = FileUtils.to_path(
            path
        )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path.write_text(
            content,
            encoding=encoding,
        )

        return file_path

    # ========================================================================
    # JSON READING
    # ========================================================================

    @staticmethod
    def read_json(
        path: str | Path,
        encoding: str = "utf-8",
    ) -> Any:
        """
        Read and parse a JSON file.

        Used primarily for:

            metadata.json

        and other project-generated JSON files.

        JSON parsing errors are allowed to propagate with
        the original exception so that the caller knows
        the file is malformed.
        """

        file_path = FileUtils.to_path(
            path
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"JSON file does not exist: "
                f"{file_path}"
            )

        if not file_path.is_file():

            raise IsADirectoryError(
                f"Expected a JSON file but "
                f"found a directory: {file_path}"
            )

        with file_path.open(
            "r",
            encoding=encoding,
        ) as file:

            return json.load(file)

    # ========================================================================
    # JSON WRITING
    # ========================================================================

    @staticmethod
    def write_json(
        path: str | Path,
        data: Any,
        encoding: str = "utf-8",
        indent: int = 2,
    ) -> Path:
        """
        Serialize data to a JSON file.

        Parent directories are created automatically.
        """

        file_path = FileUtils.to_path(
            path
        )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with file_path.open(
            "w",
            encoding=encoding,
        ) as file:

            json.dump(
                data,
                file,
                indent=indent,
                ensure_ascii=False,
            )

        return file_path

    # ========================================================================
    # JSONL APPEND
    # ========================================================================

    @staticmethod
    def append_jsonl(
        path: str | Path,
        data: Any,
        encoding: str = "utf-8",
    ) -> Path:
        """
        Append one JSON object/record to a JSONL file.

        Used by:

            replay_buffer.py

        and potentially:

            metrics_logger.py
        """

        file_path = FileUtils.to_path(
            path
        )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with file_path.open(
            "a",
            encoding=encoding,
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
            )

            file.write("\n")

        return file_path

    # ========================================================================
    # JSONL READING
    # ========================================================================

    @staticmethod
    def read_jsonl(
        path: str | Path,
        encoding: str = "utf-8",
    ) -> List[Any]:
        """
        Read all records from a JSONL file.

        Empty lines are ignored.
        """

        file_path = FileUtils.to_path(
            path
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"JSONL file does not exist: "
                f"{file_path}"
            )

        records: List[Any] = []

        with file_path.open(
            "r",
            encoding=encoding,
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:

                    records.append(
                        json.loads(line)
                    )

                except json.JSONDecodeError as exc:

                    raise ValueError(
                        f"Invalid JSONL record "
                        f"at line {line_number} "
                        f"in {file_path}: "
                        f"{exc}"
                    ) from exc

        return records

    # ========================================================================
    # FILE LISTING
    # ========================================================================

    @staticmethod
    def list_files(
        directory: str | Path,
        recursive: bool = False,
    ) -> List[Path]:
        """
        Return files contained in a directory.

        This function does NOT decide which files are relevant.

        It simply lists filesystem entries.

        File selection remains the responsibility of:

            PathResolver
        """

        directory_path = (
            FileUtils.to_path(
                directory
            )
        )

        if not directory_path.exists():

            raise FileNotFoundError(
                f"Directory does not exist: "
                f"{directory_path}"
            )

        if not directory_path.is_dir():

            raise NotADirectoryError(
                f"Expected a directory: "
                f"{directory_path}"
            )

        if recursive:

            files = [
                path
                for path in directory_path.rglob("*")
                if path.is_file()
            ]

        else:

            files = [
                path
                for path in directory_path.iterdir()
                if path.is_file()
            ]

        return sorted(files)

    # ========================================================================
    # DIRECTORY LISTING
    # ========================================================================

    @staticmethod
    def list_directories(
        directory: str | Path,
    ) -> List[Path]:
        """
        Return immediate subdirectories.

        No scenario-specific filtering is performed here.
        """

        directory_path = (
            FileUtils.to_path(
                directory
            )
        )

        if not directory_path.exists():

            raise FileNotFoundError(
                f"Directory does not exist: "
                f"{directory_path}"
            )

        if not directory_path.is_dir():

            raise NotADirectoryError(
                f"Expected a directory: "
                f"{directory_path}"
            )

        return sorted(
            [
                path
                for path in directory_path.iterdir()
                if path.is_dir()
            ]
        )

    # ========================================================================
    # FILE SIZE
    # ========================================================================

    @staticmethod
    def file_size(
        path: str | Path,
    ) -> int:
        """
        Return file size in bytes.
        """

        file_path = FileUtils.to_path(
            path
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"File does not exist: "
                f"{file_path}"
            )

        if not file_path.is_file():

            raise IsADirectoryError(
                f"Expected a file: "
                f"{file_path}"
            )

        return file_path.stat().st_size

    # ========================================================================
    # REMOVE FILE
    # ========================================================================

    @staticmethod
    def remove_file(
        path: str | Path,
    ) -> None:
        """
        Remove a file if it exists.

        This is intentionally limited to files.

        It will not recursively delete directories.
        """

        file_path = FileUtils.to_path(
            path
        )

        if not file_path.exists():
            return

        if not file_path.is_file():

            raise IsADirectoryError(
                f"Expected a file: "
                f"{file_path}"
            )

        file_path.unlink()

    # ========================================================================
    # COPY FILE
    # ========================================================================

    @staticmethod
    def copy_file(
        source: str | Path,
        destination: str | Path,
    ) -> Path:
        """
        Copy a file to another location.

        This method is kept intentionally small.

        Model/checkpoint-specific copying logic does not belong here.
        """

        import shutil

        source_path = FileUtils.to_path(
            source
        )

        destination_path = (
            FileUtils.to_path(
                destination
            )
        )

        if not source_path.exists():

            raise FileNotFoundError(
                f"Source file does not exist: "
                f"{source_path}"
            )

        if not source_path.is_file():

            raise IsADirectoryError(
                f"Source is not a file: "
                f"{source_path}"
            )

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        return destination_path