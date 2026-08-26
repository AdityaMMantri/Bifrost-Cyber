import json
from pathlib import Path
from typing import Any

from .models import Metadata


class MetadataLoader:
    """
    Loads and normalizes metadata.json for the PrimeVul builder.

    Responsibilities:
        - Read metadata.json
        - Extract relevant source files
        - Extract ground-truth vulnerability information
        - Extract CWE / attack / source / sink information
        - Preserve the original metadata
        - Validate required fields

    Important:
        metadata.json is builder-side information.
        It must NOT be directly exposed to the model.
    """

    def __init__(self, logger=None):
        self.logger = logger

    # ============================================================
    # PUBLIC API
    # ============================================================

    def load(self, metadata_path: Path) -> Metadata:
        """
        Load a metadata.json file and return a normalized Metadata
        object.
        """

        metadata_path = Path(metadata_path)

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        if not metadata_path.is_file():
            raise ValueError(
                f"Metadata path is not a file: {metadata_path}"
            )

        data = self._read_json(metadata_path)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected metadata.json to contain a JSON object: "
                f"{metadata_path}"
            )

        scenario_name = self._extract_scenario_name(
            data,
            metadata_path,
        )

        relevant_files = self._extract_file_list(
            data,
            "relevant_files",
        )

        noise_files = self._extract_file_list(
            data,
            "noise_files",
        )

        vulnerabilities = self._extract_vulnerabilities(data)

        optimal_attack = self._extract_string(
            data,
            "optimal_attack",
        )

        valid_attacks = self._extract_string_list(
            data,
            "valid_attacks",
        )

        cwe_reference = self._extract_cwe(data)

        injection_point = self._extract_string(
            data,
            "injection_point",
        )

        sink = self._extract_string(
            data,
            "sink",
        )

        severity = self._extract_string(
            data,
            "severity",
        )

        description = self._extract_description(data)

        requires_cross_file_reasoning = self._extract_bool(
            data,
            "requires_cross_file_reasoning",
        )

        metadata = Metadata(
            scenario_name=scenario_name,
            relevant_files=relevant_files,
            noise_files=noise_files,
            technology_stack=self._extract_string_list(
                data,
                "technology_stack",
            ),
            trust_boundary=self._extract_trust_boundary(data),
            vulnerabilities=vulnerabilities,
            optimal_attack=optimal_attack,
            valid_attacks=valid_attacks,
            cwe_reference=cwe_reference,
            injection_point=injection_point,
            sink=sink,
            severity=severity,
            description=description,
            requires_cross_file_reasoning=(
                requires_cross_file_reasoning
            ),
            raw_data=data,
        )

        self._validate(metadata, metadata_path)

        if self.logger:
            self.logger.info(
                f"Loaded metadata: {metadata.scenario_name}"
            )
            self.logger.info(
                f"Relevant files: {len(metadata.relevant_files)}"
            )

        return metadata

    # ============================================================
    # JSON READING
    # ============================================================

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in metadata file: {path}\n"
                f"{exc}"
            ) from exc

        except OSError as exc:
            raise OSError(
                f"Could not read metadata file: {path}\n"
                f"{exc}"
            ) from exc

    # ============================================================
    # SCENARIO NAME
    # ============================================================

    def _extract_scenario_name(
        self,
        data: dict[str, Any],
        metadata_path: Path,
    ) -> str:

        possible_keys = [
            "scenario_name",
            "scenario_id",
            "id",
            "name",
            "scenario",
        ]

        for key in possible_keys:
            value = data.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        # Fallback to scenario directory name.
        return metadata_path.parent.name

    # ============================================================
    # FILE LISTS
    # ============================================================

    def _extract_file_list(
        self,
        data: dict[str, Any],
        key: str,
    ) -> list[str]:

        value = data.get(key, [])

        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError(
                f"'{key}' must be a list, "
                f"got {type(value).__name__}"
            )

        result = []

        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())

        return result

    # ============================================================
    # VULNERABILITIES
    # ============================================================

    def _extract_vulnerabilities(
        self,
        data: dict[str, Any],
    ) -> list[dict]:

        value = data.get("vulnerabilities", [])

        if value is None:
            return []

        if isinstance(value, dict):
            return [value]

        if not isinstance(value, list):
            return []

        return [
            item
            for item in value
            if isinstance(item, dict)
        ]

    # ============================================================
    # STRING VALUES
    # ============================================================

    def _extract_string(
        self,
        data: dict[str, Any],
        key: str,
    ) -> str | None:

        value = data.get(key)

        if isinstance(value, str):
            value = value.strip()

            if value:
                return value

        return None

    # ============================================================
    # STRING LISTS
    # ============================================================

    def _extract_string_list(
        self,
        data: dict[str, Any],
        key: str,
    ) -> list[str]:

        value = data.get(key, [])

        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if not isinstance(value, list):
            return []

        return [
            item.strip()
            for item in value
            if isinstance(item, str)
            and item.strip()
        ]

    # ============================================================
    # CWE
    # ============================================================

    def _extract_cwe(
        self,
        data: dict[str, Any],
    ) -> str | None:

        value = data.get("cwe_reference")

        if isinstance(value, str):
            return value.strip() or None

        # Some metadata schemas may store CWE information
        # under a vulnerability object.
        vulnerabilities = data.get(
            "vulnerabilities",
            [],
        )

        if isinstance(vulnerabilities, list):

            for vulnerability in vulnerabilities:

                if not isinstance(vulnerability, dict):
                    continue

                for key in (
                    "cwe",
                    "cwe_id",
                    "cwe_reference",
                ):
                    cwe = vulnerability.get(key)

                    if isinstance(cwe, str):
                        cwe = cwe.strip()

                        if cwe:
                            return cwe

        return None

    # ============================================================
    # DESCRIPTION
    # ============================================================

    def _extract_description(
        self,
        data: dict[str, Any],
    ) -> str | None:

        direct_description = self._extract_string(
            data,
            "description",
        )

        if direct_description:
            return direct_description

        vulnerabilities = data.get(
            "vulnerabilities",
            [],
        )

        if isinstance(vulnerabilities, list):

            for vulnerability in vulnerabilities:

                if not isinstance(vulnerability, dict):
                    continue

                for key in (
                    "description",
                    "vulnerability_description",
                    "summary",
                ):
                    description = vulnerability.get(key)

                    if isinstance(description, str):
                        description = description.strip()

                        if description:
                            return description

        return None

    # ============================================================
    # TRUST BOUNDARY
    # ============================================================

    def _extract_trust_boundary(
        self,
        data: dict[str, Any],
    ) -> str | None:

        value = data.get("trust_boundary")

        if isinstance(value, str):
            return value.strip() or None

        # Handle schemas where security information is nested.
        security = data.get("security")

        if isinstance(security, dict):

            value = security.get("trust_boundary")

            if isinstance(value, str):
                return value.strip() or None

        return None

    # ============================================================
    # BOOLEAN
    # ============================================================

    def _extract_bool(
        self,
        data: dict[str, Any],
        key: str,
    ) -> bool:

        value = data.get(key)

        if isinstance(value, bool):
            return value

        return False

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate(
        self,
        metadata: Metadata,
        metadata_path: Path,
    ) -> None:

        if not metadata.relevant_files:
            raise ValueError(
                f"No relevant_files found in "
                f"{metadata_path}"
            )

        # Relevant and noise files should not overlap.
        overlap = set(
            metadata.relevant_files
        ).intersection(
            metadata.noise_files
        )

        if overlap:
            raise ValueError(
                f"Files appear in both relevant_files and "
                f"noise_files in {metadata_path}: "
                f"{sorted(overlap)}"
            )

    # ============================================================
    # CONVENIENCE METHODS
    # ============================================================

    def get_relevant_files(
        self,
        metadata: Metadata,
    ) -> list[str]:
        """
        Return files that should become the model's primary
        source-code input.
        """

        return list(metadata.relevant_files)

    def get_noise_files(
        self,
        metadata: Metadata,
    ) -> list[str]:
        """
        Return noise files.

        These are retained for metadata/debugging purposes but
        should NOT be included in the PrimeVul-style source
        input.
        """

        return list(metadata.noise_files)

    def get_ground_truth_vulnerability(
        self,
        metadata: Metadata,
    ) -> dict | None:
        """
        Return the first available vulnerability record.

        This is builder-side information only.
        """

        if not metadata.vulnerabilities:
            return None

        return metadata.vulnerabilities[0]