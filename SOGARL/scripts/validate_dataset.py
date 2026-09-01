"""
validate_dataset.py

Validate the SOGARL SFT dataset before training.

Checks:
    - dataset directory exists
    - first 100 scenario directories exist
    - metadata file exists and is valid
    - scenario.md exists
    - files/ directory exists
    - metadata contains required file information
    - all relevant files exist
    - noise files are valid
    - duplicate / invalid paths are detected

This script validates the dataset only.
It does not:
    - modify files
    - build prompts
    - load model adapters
    - generate responses
    - run Oracle
    - perform training
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from configs import config
from src.utils.file_utils import FileUtils


# ============================================================================
# CONFIGURATION
# ============================================================================

METADATA_FILENAME = "metadata.json"
SCENARIO_FILENAME = "scenario.md"
CODE_DIRECTORY = "files"

# Validate only scenario_001 through scenario_100.
MAX_SCENARIO = 100


# ============================================================================
# VALIDATION RESULT
# ============================================================================

class ValidationResult:

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.scenarios_checked = 0

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


# ============================================================================
# DATASET VALIDATOR
# ============================================================================

class DatasetValidator:

    def __init__(
        self,
        dataset_path: Path,
        logger=None,
    ):
        self.dataset_path = Path(dataset_path)
        self.logger = logger
        self.result = ValidationResult()

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def validate(self) -> ValidationResult:
        """Validate the first 100 scenarios."""

        self._validate_dataset_root()

        if not self.dataset_path.exists():
            return self.result

        scenarios = self._find_scenarios()

        if not scenarios:
            self.result.error(
                f"No scenario directories found in: "
                f"{self.dataset_path}"
            )
            return self.result

        for scenario_path in scenarios:
            self._validate_scenario(
                scenario_path
            )

        return self.result

    # ========================================================================
    # DATASET ROOT
    # ========================================================================

    def _validate_dataset_root(self) -> None:

        if not self.dataset_path.exists():

            self.result.error(
                f"Dataset path does not exist: "
                f"{self.dataset_path}"
            )

            return

        if not self.dataset_path.is_dir():

            self.result.error(
                f"Dataset path is not a directory: "
                f"{self.dataset_path}"
            )

    # ========================================================================
    # FIND SCENARIOS
    # ========================================================================

    def _find_scenarios(self) -> List[Path]:
        """
        Find only scenario_001 through scenario_100.

        The numeric portion of the directory name is used, so formats such
        as scenario_1 and scenario_001 are both accepted.
        """

        scenarios: List[Path] = []

        for path in self.dataset_path.iterdir():

            if not path.is_dir():
                continue

            name = path.name.strip()

            if not name.lower().startswith(
                "scenario_"
            ):
                continue

            suffix = name.split(
                "_",
                1,
            )[1]

            try:
                scenario_number = int(
                    suffix
                )
            except (ValueError, TypeError):
                continue

            if 1 <= scenario_number <= MAX_SCENARIO:
                scenarios.append(path)

        return sorted(
            scenarios,
            key=lambda path: self._directory_scenario_number(
                path.name
            )
        )

    # ========================================================================
    # SCENARIO NUMBER
    # ========================================================================

    @staticmethod
    def _directory_scenario_number(
        scenario_name: str,
    ) -> int:

        try:

            return int(
                scenario_name.split(
                    "_",
                    1,
                )[1]
            )

        except (
            ValueError,
            IndexError,
        ):

            return 0

    @staticmethod
    def _normalize_scenario_id(
        value: Any,
    ) -> Optional[int]:
        """
        Normalize different scenario_id representations.

        Accepted examples:

            1
            01
            001
            "1"
            "01"
            "001"
            "scenario_1"
            "scenario_01"
            "scenario_001"
            "SCENARIO_001"

        All of the above normalize to:

            1
        """

        if value is None:
            return None

        text = str(
            value
        ).strip()

        if not text:
            return None

        text = text.lower()

        if text.startswith(
            "scenario_"
        ):

            text = text[
                len("scenario_"):
            ]

        try:

            return int(
                text
            )

        except (
            ValueError,
            TypeError,
        ):

            return None

    # ========================================================================
    # SCENARIO
    # ========================================================================

    def _validate_scenario(
        self,
        scenario_path: Path,
    ) -> None:

        self.result.scenarios_checked += 1

        scenario_name = (
            scenario_path.name
        )

        self._log(
            f"Checking {scenario_name}"
        )

        metadata_path = (
            scenario_path
            / METADATA_FILENAME
        )

        scenario_md_path = (
            scenario_path
            / SCENARIO_FILENAME
        )

        files_path = (
            scenario_path
            / CODE_DIRECTORY
        )

        # --------------------------------------------------------------------
        # Required files/directories
        # --------------------------------------------------------------------

        if not metadata_path.is_file():

            self.result.error(
                f"{scenario_name}: missing "
                f"{METADATA_FILENAME}"
            )

            return

        if not scenario_md_path.is_file():

            self.result.error(
                f"{scenario_name}: missing "
                f"{SCENARIO_FILENAME}"
            )

        if not files_path.is_dir():

            self.result.error(
                f"{scenario_name}: missing "
                f"{CODE_DIRECTORY}/ directory"
            )

            return

        # --------------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------------

        metadata = self._load_metadata(
            metadata_path
        )

        if metadata is None:
            return

        self._validate_metadata(
            scenario_name=scenario_name,
            metadata=metadata,
            files_path=files_path,
        )

    # ========================================================================
    # LOAD METADATA
    # ========================================================================

    def _load_metadata(
        self,
        metadata_path: Path,
    ) -> Optional[Dict[str, Any]]:

        try:

            metadata = FileUtils.read_json(
                metadata_path
            )

        except Exception as exc:

            self.result.error(
                f"{metadata_path.parent.name}: "
                f"could not read metadata.json: "
                f"{exc}"
            )

            return None

        if not isinstance(
            metadata,
            dict,
        ):

            self.result.error(
                f"{metadata_path.parent.name}: "
                f"metadata.json must contain "
                f"a JSON object"
            )

            return None

        return metadata

    # ========================================================================
    # METADATA VALIDATION
    # ========================================================================

    def _validate_metadata(
        self,
        scenario_name: str,
        metadata: Dict[str, Any],
        files_path: Path,
    ) -> None:

        required_fields = [
            "scenario_id",
            "relevant_files",
        ]

        for field_name in required_fields:

            if field_name not in metadata:

                self.result.error(
                    f"{scenario_name}: metadata.json "
                    f"missing required field "
                    f"'{field_name}'"
                )

        # --------------------------------------------------------------------
        # Scenario ID
        #
        # Formatting differences are allowed.
        #
        # Examples:
        #
        #     scenario_001 + 001       -> valid
        #     scenario_001 + "001"     -> valid
        #     scenario_001 + 1         -> valid
        #     scenario_001 + "1"       -> valid
        #     scenario_001 + scenario_001 -> valid
        #
        # Only an actual numeric mismatch is an error.
        # --------------------------------------------------------------------

        scenario_id = metadata.get(
            "scenario_id"
        )

        if scenario_id is not None:

            expected_number = (
                self._directory_scenario_number(
                    scenario_name
                )
            )

            actual_number = (
                self._normalize_scenario_id(
                    scenario_id
                )
            )

            if actual_number is None:

                self.result.error(
                    f"{scenario_name}: invalid "
                    f"scenario_id='{scenario_id}'"
                )

            elif actual_number != expected_number:

                self.result.error(
                    f"{scenario_name}: metadata "
                    f"scenario_id='{scenario_id}' "
                    f"does not match directory "
                    f"scenario number "
                    f"'{expected_number}'"
                )

        relevant_files = metadata.get(
            "relevant_files",
            [],
        )

        noise_files = metadata.get(
            "noise_files",
            [],
        )

        self._validate_file_list(
            scenario_name=scenario_name,
            field_name="relevant_files",
            file_list=relevant_files,
            files_path=files_path,
            required=True,
        )

        self._validate_file_list(
            scenario_name=scenario_name,
            field_name="noise_files",
            file_list=noise_files,
            files_path=files_path,
            required=False,
        )

        self._validate_noise_count(
            scenario_name=scenario_name,
            noise_files=noise_files,
        )

    # ========================================================================
    # FILE PATH NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_file_path(
        relative_path: str,
    ) -> str:
        """
        Normalize metadata file paths.

        Both are accepted:

            files/identity/TrustPolicy.php
            identity/TrustPolicy.php

        because files_path already points to the scenario's files/ directory.
        """

        normalized = (
            relative_path
            .replace("\\", "/")
            .strip()
        )

        while normalized.startswith("./"):
            normalized = normalized[2:]

        if normalized == "files":

            return ""

        if normalized.startswith(
            "files/"
        ):

            normalized = normalized[
                len("files/"):
            ]

        return normalized

    # ========================================================================
    # CASE-INSENSITIVE FILE RESOLUTION
    # ========================================================================

    @staticmethod
    def _resolve_case_insensitive_file(
        files_path: Path,
        normalized_path: str,
    ) -> Optional[Path]:
        """
        Resolve a metadata-defined file path.

        First performs an exact filesystem lookup.

        If the exact path does not exist, performs a case-insensitive
        lookup using the complete relative path.

        This is required because the dataset was created/validated on
        Windows, which is normally case-insensitive, while Kaggle runs
        Linux, which is case-sensitive.

        Returns
        -------
        Optional[Path]
            The actual filesystem path if exactly one match exists.
            None if no match exists.

        Raises
        ------
        ValueError
            If multiple case-insensitive matches exist.
        """

        exact_path = (
            files_path
            / Path(normalized_path)
        ).resolve()

        if exact_path.is_file():
            return exact_path

        target_parts = Path(
            normalized_path
        ).parts

        matches: List[Path] = []

        for candidate in files_path.rglob("*"):

            if not candidate.is_file():
                continue

            try:

                candidate_parts = (
                    candidate
                    .relative_to(files_path)
                    .parts
                )

            except ValueError:

                continue

            if len(candidate_parts) != len(
                target_parts
            ):
                continue

            if all(
                actual.lower() == expected.lower()
                for actual, expected
                in zip(
                    candidate_parts,
                    target_parts,
                )
            ):

                matches.append(
                    candidate.resolve()
                )

        if len(matches) == 1:

            return matches[0]

        if len(matches) > 1:

            match_text = "\n".join(
                f"  - {match}"
                for match in matches
            )

            raise ValueError(
                "Ambiguous case-insensitive file match:\n"
                f"Metadata path: {normalized_path}\n"
                f"Matches:\n"
                f"{match_text}"
            )

        return None

    # ========================================================================
    # FILE LIST VALIDATION
    # ========================================================================

    def _validate_file_list(
        self,
        scenario_name: str,
        field_name: str,
        file_list: Any,
        files_path: Path,
        required: bool,
    ) -> None:

        if file_list is None:
            file_list = []

        if not isinstance(
            file_list,
            list,
        ):

            self.result.error(
                f"{scenario_name}: "
                f"'{field_name}' must be a list"
            )

            return

        if required and not file_list:

            self.result.error(
                f"{scenario_name}: "
                f"'{field_name}' cannot be empty"
            )

            return

        seen = set()

        for relative_path in file_list:

            if not isinstance(
                relative_path,
                str,
            ):

                self.result.error(
                    f"{scenario_name}: "
                    f"'{field_name}' contains "
                    f"a non-string path"
                )

                continue

            normalized_path = (
                self._normalize_file_path(
                    relative_path
                )
            )

            if not normalized_path:

                self.result.error(
                    f"{scenario_name}: invalid "
                    f"empty path in '{field_name}'"
                )

                continue

            # ---------------------------------------------------------------
            # Duplicate detection is performed after normalization.
            #
            # Case differences are treated as the same logical path.
            # This matches the cross-platform resolution behavior.
            # ---------------------------------------------------------------

            duplicate_key = (
                normalized_path.lower()
            )

            if duplicate_key in seen:

                self.result.error(
                    f"{scenario_name}: duplicate "
                    f"path in '{field_name}': "
                    f"{relative_path}"
                )

                continue

            seen.add(
                duplicate_key
            )

            # ---------------------------------------------------------------
            # Security check
            # ---------------------------------------------------------------

            file_path = (
                files_path
                / Path(normalized_path)
            )

            try:

                file_path.resolve().relative_to(
                    files_path.resolve()
                )

            except ValueError:

                self.result.error(
                    f"{scenario_name}: invalid "
                    f"path outside files/: "
                    f"{relative_path}"
                )

                continue

            # ---------------------------------------------------------------
            # Existence check
            #
            # First try the exact path. If it is not present, resolve the
            # metadata path case-insensitively so datasets created on
            # Windows work correctly on Linux/Kaggle.
            # ---------------------------------------------------------------

            try:

                resolved_path = (
                    self._resolve_case_insensitive_file(
                        files_path=files_path,
                        normalized_path=normalized_path,
                    )
                )

            except ValueError as exc:

                self.result.error(
                    f"{scenario_name}: "
                    f"{field_name} has an ambiguous "
                    f"file path '{relative_path}': "
                    f"{exc}"
                )

                continue

            if resolved_path is None:

                self.result.error(
                    f"{scenario_name}: file listed in "
                    f"'{field_name}' does not exist: "
                    f"{relative_path}"
                )

    # ========================================================================
    # NOISE VALIDATION
    # ========================================================================

    def _validate_noise_count(
        self,
        scenario_name: str,
        noise_files: Any,
    ) -> None:

        if not isinstance(
            noise_files,
            list,
        ):
            return

        noise_count = len(
            noise_files
        )

        if noise_count < 2:

            self.result.warning(
                f"{scenario_name}: only "
                f"{noise_count} noise file(s) available. "
                f"At least 2 are recommended."
            )

        # This is only informational. It is not an error.
        if noise_count > 5:

            self.result.warning(
                f"{scenario_name}: "
                f"{noise_count} noise files listed. "
                f"PathResolver will limit runtime "
                f"noise selection according to its configuration."
            )

    # ========================================================================
    # LOGGING
    # ========================================================================

    def _log(
        self,
        message: str,
    ) -> None:

        if self.logger:

            self.logger.info(
                message
            )


# ============================================================================
# REPORT
# ============================================================================

def print_report(
    result: ValidationResult,
) -> None:

    print()
    print("=" * 70)
    print("SOGARL DATASET VALIDATION")
    print("=" * 70)

    print(
        f"Scenarios checked : "
        f"{result.scenarios_checked}"
    )

    print(
        f"Errors            : "
        f"{len(result.errors)}"
    )

    print(
        f"Warnings          : "
        f"{len(result.warnings)}"
    )

    print()

    if result.errors:

        print("ERRORS")
        print("-" * 70)

        for error in result.errors:

            print(
                f"[ERROR] {error}"
            )

        print()

    if result.warnings:

        print("WARNINGS")
        print("-" * 70)

        for warning in result.warnings:

            print(
                f"[WARNING] {warning}"
            )

        print()

    print("RESULT")
    print("-" * 70)

    if result.valid:

        print(
            "Dataset validation PASSED."
        )

    else:

        print(
            "Dataset validation FAILED."
        )

    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    dataset_path = Path(
        config.DATASET_PATH
    )

    validator = DatasetValidator(
        dataset_path=dataset_path,
    )

    result = validator.validate()

    print_report(
        result
    )

    return (
        0
        if result.valid
        else 1
    )


if __name__ == "__main__":

    sys.exit(
        main()
    )