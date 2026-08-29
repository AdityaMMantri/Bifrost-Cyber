from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Metadata


class MetadataLoader:
    """
    Loads metadata.json for the PrimeVul builder.

    IMPORTANT
    ---------
    metadata.json is builder-side ground truth.

    It is NEVER passed directly to the classification model.

    This loader is responsible for:
        - reading metadata.json
        - normalizing metadata fields
        - extracting relevant/noise files
        - extracting vulnerability information
        - preserving ground-truth information for builder-side use

    It does NOT load scenario.md.
    scenario.md is handled exclusively by ScenarioLoader.
    """

    def __init__(self, logger=None):
        self.logger = logger

    # ============================================================
    # PUBLIC API
    # ============================================================

    def load(
        self,
        metadata_path: Path,
    ) -> Metadata:
        """
        Load and normalize one metadata.json file.
        """

        metadata_path = Path(
            metadata_path
        )

        if not metadata_path.exists():

            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{metadata_path}"
            )

        if not metadata_path.is_file():

            raise ValueError(
                f"Metadata path is not a file: "
                f"{metadata_path}"
            )

        data = self._read_json(
            metadata_path
        )

        if not isinstance(
            data,
            dict,
        ):

            raise ValueError(
                "Expected metadata.json to contain "
                f"a JSON object: {metadata_path}"
            )

        # ========================================================
        # SCENARIO ID / NAME
        # ========================================================

        scenario_id = (
            self._extract_string(
                data,
                "scenario_id",
            )
        )

        scenario_name = (
            self._extract_scenario_name(
                data,
                metadata_path,
            )
        )

        # ========================================================
        # FILE LISTS
        # ========================================================

        relevant_files = (
            self._extract_file_list(
                data,
                "relevant_files",
            )
        )

        noise_files = (
            self._extract_file_list(
                data,
                "noise_files",
            )
        )

        # ========================================================
        # VULNERABILITIES
        # ========================================================

        vulnerabilities = (
            self._extract_vulnerabilities(
                data
            )
        )

        # ========================================================
        # CLASSIFICATION
        # ========================================================

        classification = data.get(
            "classification"
        )

        if not isinstance(
            classification,
            dict,
        ):

            classification = {}

        # ========================================================
        # BUILD METADATA OBJECT
        #
        # IMPORTANT FIX:
        #
        # Metadata defines:
        #
        #     name: Optional[str]
        #
        # and scenario_name is a property.
        #
        # Therefore:
        #
        #     name=scenario_name
        #
        # NOT:
        #
        #     scenario_name=scenario_name
        # ========================================================

        metadata = Metadata(
    scenario_id=self._extract_string(
        data,
        "scenario_id",
    ),

    name=scenario_name,

    category=self._extract_string(
        data,
        "category",
    ),

            attack_type=self._extract_string(
                data,
                "attack_type",
            ),

            difficulty=self._extract_string(
                data,
                "difficulty",
            ),

            scenario_type=self._extract_string(
                data,
                "scenario_type",
            ),

            vulnerability_count=self._extract_int(
                data,
                "vulnerability_count",
                len(vulnerabilities),
            ),

            vulnerabilities=vulnerabilities,

            optimal_attack=self._extract_string(
                data,
                "optimal_attack",
            ),

            optimal_defense=self._extract_string(
                data,
                "optimal_defense",
            ),

            valid_attacks=self._extract_string_list(
                data,
                "valid_attacks",
            ),

            valid_defenses=self._extract_string_list(
                data,
                "valid_defenses",
            ),

            cwe_reference=self._extract_cwe(
                data
            ),

            cve=self._extract_string(
                data,
                "cve",
            )
            or self._extract_string(
                data,
                "cve_reference",
            ),

            cve_desc=self._extract_string(
                data,
                "cve_desc",
            ),

            source=self._extract_string(
                data,
                "source",
            ),

            sink=self._extract_string(
                data,
                "sink",
            ),

            injection_point=self._extract_string(
                data,
                "injection_point",
            ),

            severity=self._extract_string(
                data,
                "severity",
            ),

            description=self._extract_description(
                data
            ),

            trust_boundary=self._extract_trust_boundary(
                data
            ),

            requires_cross_file_reasoning=(
                self._extract_bool(
                    data,
                    "requires_cross_file_reasoning",
                )
            ),

            reasoning_depth=self._extract_int(
                data,
                "reasoning_depth",
                None,
            ),

            competing_hypotheses=data.get(
                "competing_hypotheses",
                [],
            ),

            tags=self._extract_string_list(
                data,
                "tags",
            ),

            classification=classification,

            relevant_files=relevant_files,

            noise_files=noise_files,

            raw_data=data,
        )

        # ========================================================
        # NORMALIZE COMPETING HYPOTHESES
        # ========================================================

        if not isinstance(
            metadata.competing_hypotheses,
            list,
        ):

            metadata.competing_hypotheses = []

        # ========================================================
        # VALIDATE
        # ========================================================

        self._validate(
            metadata,
            metadata_path,
        )

        # ========================================================
        # LOGGING
        # ========================================================

        if self.logger:

            self.logger.info(
                f"Loaded metadata: "
                f"{metadata.scenario_name}"
            )

            self.logger.info(
                f"Scenario ID: "
                f"{metadata.scenario_id}"
            )

            self.logger.info(
                f"Relevant files: "
                f"{len(metadata.relevant_files)}"
            )

            self.logger.info(
                f"Noise files: "
                f"{len(metadata.noise_files)}"
            )

            self.logger.info(
                f"Vulnerability count: "
                f"{metadata.vulnerability_count}"
            )

        return metadata

    # ============================================================
    # JSON READING
    # ============================================================

    def _read_json(
        self,
        path: Path,
    ) -> dict[str, Any]:

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(
                    file
                )

        except json.JSONDecodeError as exc:

            raise ValueError(
                f"Invalid JSON in metadata file: "
                f"{path}\n{exc}"
            ) from exc

        except OSError as exc:

            raise OSError(
                f"Could not read metadata file: "
                f"{path}\n{exc}"
            ) from exc

    # ============================================================
    # SCENARIO NAME
    # ============================================================

    def _extract_scenario_name(
        self,
        data: dict[str, Any],
        metadata_path: Path,
    ) -> str:

        for key in (
            "scenario_name",
            "name",
            "scenario_id",
            "id",
            "scenario",
        ):

            value = data.get(
                key
            )

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):

                return value.strip()

        # Final fallback:
        # directory name, e.g. scenario_019

        return metadata_path.parent.name

    # ============================================================
    # FILE LIST
    # ============================================================

    def _extract_file_list(
        self,
        data: dict[str, Any],
        key: str,
    ) -> list[str]:

        value = data.get(
            key,
            [],
        )

        if value is None:

            return []

        if not isinstance(
            value,
            list,
        ):

            raise ValueError(
                f"'{key}' must be a list, "
                f"got {type(value).__name__}"
            )

        result: list[str] = []

        for item in value:

            if (
                isinstance(
                    item,
                    str,
                )
                and item.strip()
            ):

                result.append(
                    item.strip()
                )

        return result

    # ============================================================
    # VULNERABILITIES
    # ============================================================

    def _extract_vulnerabilities(
        self,
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:

        value = data.get(
            "vulnerabilities",
            [],
        )

        if value is None:

            return []

        if isinstance(
            value,
            dict,
        ):

            return [value]

        if not isinstance(
            value,
            list,
        ):

            return []

        return [
            item
            for item in value
            if isinstance(
                item,
                dict,
            )
        ]

    # ============================================================
    # STRING
    # ============================================================

    def _extract_string(
        self,
        data: dict[str, Any],
        key: str,
    ) -> str | None:

        value = data.get(
            key
        )

        if (
            isinstance(
                value,
                str,
            )
            and value.strip()
        ):

            return value.strip()

        return None

    # ============================================================
    # STRING LIST
    # ============================================================

    def _extract_string_list(
        self,
        data: dict[str, Any],
        key: str,
    ) -> list[str]:

        value = data.get(
            key,
            [],
        )

        if value is None:

            return []

        if isinstance(
            value,
            str,
        ):

            value = [
                value
            ]

        if not isinstance(
            value,
            list,
        ):

            return []

        return [
            item.strip()
            for item in value
            if (
                isinstance(
                    item,
                    str,
                )
                and item.strip()
            )
        ]

    # ============================================================
    # INTEGER
    # ============================================================

    def _extract_int(
        self,
        data: dict[str, Any],
        key: str,
        default,
    ):

        value = data.get(
            key
        )

        if (
            isinstance(
                value,
                int,
            )
            and not isinstance(
                value,
                bool,
            )
        ):

            return value

        return default

    # ============================================================
    # BOOLEAN
    # ============================================================

    def _extract_bool(
        self,
        data: dict[str, Any],
        key: str,
    ) -> bool:

        value = data.get(
            key
        )

        if isinstance(
            value,
            bool,
        ):

            return value

        return False

    # ============================================================
    # CWE
    # ============================================================

    def _extract_cwe(
        self,
        data: dict[str, Any],
    ):

        value = data.get(
            "cwe_reference",
            data.get(
                "cwe"
            ),
        )

        if isinstance(
            value,
            (
                str,
                list,
            ),
        ):

            return value

        # --------------------------------------------------------
        # Fallback to vulnerability-level CWE
        # --------------------------------------------------------

        for vulnerability in (
            self._extract_vulnerabilities(
                data
            )
        ):

            for key in (
                "cwe",
                "cwe_id",
                "cwe_reference",
            ):

                cwe = vulnerability.get(
                    key
                )

                if isinstance(
                    cwe,
                    (
                        str,
                        list,
                    ),
                ):

                    return cwe

        return None

    # ============================================================
    # DESCRIPTION
    # ============================================================

    def _extract_description(
        self,
        data: dict[str, Any],
    ) -> str | None:

        # --------------------------------------------------------
        # Top-level description
        #
        # IMPORTANT:
        # This remains builder-side metadata.
        # It must NOT be inserted into the model prompt.
        # --------------------------------------------------------

        description = (
            self._extract_string(
                data,
                "description",
            )
        )

        if description:

            return description

        # --------------------------------------------------------
        # Vulnerability-level description
        # --------------------------------------------------------

        for vulnerability in (
            self._extract_vulnerabilities(
                data
            )
        ):

            for key in (
                "description",
                "vulnerability_description",
                "summary",
            ):

                value = vulnerability.get(
                    key
                )

                if (
                    isinstance(
                        value,
                        str,
                    )
                    and value.strip()
                ):

                    return value.strip()

        return None

    # ============================================================
    # TRUST BOUNDARY
    # ============================================================

    def _extract_trust_boundary(
        self,
        data: dict[str, Any],
    ) -> str | None:

        value = (
            self._extract_string(
                data,
                "trust_boundary",
            )
        )

        if value:

            return value

        security = data.get(
            "security"
        )

        if isinstance(
            security,
            dict,
        ):

            return self._extract_string(
                security,
                "trust_boundary",
            )

        # --------------------------------------------------------
        # Some scenarios use trust_boundaries plural.
        # Keep this as builder-side metadata.
        # --------------------------------------------------------

        boundaries = data.get(
            "trust_boundaries"
        )

        if isinstance(
            boundaries,
            list,
        ):

            values = [
                str(item).strip()
                for item in boundaries
                if (
                    isinstance(
                        item,
                        str,
                    )
                    and item.strip()
                )
            ]

            if values:

                return "\n".join(
                    values
                )

        return None

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate(
        self,
        metadata: Metadata,
        metadata_path: Path,
    ) -> None:

        # --------------------------------------------------------
        # Relevant files are mandatory for PrimeVul samples.
        # --------------------------------------------------------

        if not metadata.relevant_files:

            raise ValueError(
                f"No relevant_files found in "
                f"{metadata_path}"
            )

        # --------------------------------------------------------
        # Relevant/noise overlap is invalid.
        # --------------------------------------------------------

        overlap = set(
            metadata.relevant_files
        ).intersection(
            metadata.noise_files
        )

        if overlap:

            raise ValueError(
                "Files appear in both "
                "relevant_files and noise_files "
                f"in {metadata_path}: "
                f"{sorted(overlap)}"
            )

        # --------------------------------------------------------
        # Vulnerability count consistency
        #
        # Do not silently rewrite the supplied metadata.
        # Warn if it differs from the actual vulnerability
        # object count.
        # --------------------------------------------------------

        actual_count = len(
            metadata.vulnerabilities
        )

        if (
            metadata.vulnerability_count
            != actual_count
        ):

            # A zero vulnerability count with no vulnerability
            # objects is perfectly valid for safe scenarios.
            #
            # For inconsistent vulnerable metadata, log a warning
            # rather than changing ground truth here.

            if self.logger:

                self.logger.warning(
                    f"{metadata.scenario_name}: "
                    f"metadata vulnerability_count="
                    f"{metadata.vulnerability_count}, "
                    f"but parsed vulnerabilities="
                    f"{actual_count}"
                )

    # ============================================================
    # ACCESSORS
    # ============================================================

    def get_relevant_files(
        self,
        metadata: Metadata,
    ) -> list[str]:

        return list(
            metadata.relevant_files
        )

    def get_noise_files(
        self,
        metadata: Metadata,
    ) -> list[str]:

        return list(
            metadata.noise_files
        )

    def get_ground_truth_vulnerability(
        self,
        metadata: Metadata,
    ) -> dict | None:

        if not metadata.vulnerabilities:

            return None

        return metadata.vulnerabilities[
            0
        ]