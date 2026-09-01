"""
metadata_loader.py

Loads metadata.json for SOGARL.

Responsibilities
----------------
MetadataLoader is responsible for:

    - Loading metadata.json
    - Validating that the JSON is valid
    - Accepting supported metadata schema variations
    - Normalizing metadata into a consistent SOGARL interface
    - Preserving the complete original metadata
    - Providing convenient access to fields required by SOGARL

MetadataLoader does NOT:

    - Load scenario.md
    - Load source-code files
    - Select relevant files
    - Select noise files
    - Build prompts
    - Evaluate Red or Blue responses
    - Calculate rewards
    - Create defense mappings
    - Modify the original metadata

Important
---------
The complete original metadata is preserved in `raw_data`.

This allows the Oracle and other components to access fields
that may not currently be explicitly used by SOGARL.

The loader supports multiple metadata schemas while exposing
one stable normalized interface to the rest of SOGARL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class MetadataLoader:

    def __init__(self, logger=None):
        self.logger = logger

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def load(
        self,
        metadata_path: Path | str,
    ) -> Dict[str, Any]:
        """
        Load one metadata.json file.

        Parameters
        ----------
        metadata_path:
            Path to metadata.json.

        Returns
        -------
        Dict[str, Any]
            Normalized metadata dictionary.

        The original JSON is preserved under:

            raw_data
        """

        path = (
            Path(metadata_path)
            .expanduser()
            .resolve()
        )

        self._validate_path(
            path
        )

        data = self._read_json(
            path
        )

        self._validate_root(
            data,
            path,
        )

        return self._normalize_metadata(
            data,
            path,
        )

    # ==================================================================
    # ALIAS
    # ==================================================================

    def load_metadata(
        self,
        metadata_path: Path | str,
    ) -> Dict[str, Any]:
        """
        Alias for load().

        Kept for compatibility with callers that use
        load_metadata().
        """

        return self.load(
            metadata_path
        )

    # ==================================================================
    # READ JSON
    # ==================================================================

    @staticmethod
    def _read_json(
        path: Path,
    ) -> Dict[str, Any]:
        """
        Read metadata.json.
        """

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
                "Invalid JSON in metadata file:\n"
                f"{path}\n"
                f"Line: {exc.lineno}, "
                f"Column: {exc.colno}\n"
                f"{exc.msg}"
            ) from exc

        except OSError as exc:

            raise OSError(
                "Could not read metadata file:\n"
                f"{path}\n"
                f"{exc}"
            ) from exc

    # ==================================================================
    # PATH VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_path(
        path: Path,
    ) -> None:
        """
        Validate metadata.json path.
        """

        if not path.exists():

            raise FileNotFoundError(
                f"Metadata file does not exist:\n"
                f"{path}"
            )

        if not path.is_file():

            raise ValueError(
                f"Metadata path is not a file:\n"
                f"{path}"
            )

        if path.name.lower() != "metadata.json":

            raise ValueError(
                "Expected metadata.json but received:\n"
                f"{path.name}"
            )

    # ==================================================================
    # ROOT VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_root(
        data: Any,
        path: Path,
    ) -> None:
        """
        metadata.json must contain a JSON object.
        """

        if not isinstance(
            data,
            dict,
        ):

            raise ValueError(
                "metadata.json must contain "
                "a JSON object at the root:\n"
                f"{path}"
            )

    # ==================================================================
    # NORMALIZATION
    # ==================================================================

    def _normalize_metadata(
        self,
        data: Dict[str, Any],
        metadata_path: Path,
    ) -> Dict[str, Any]:
        """
        Normalize metadata while preserving everything.

        Different scenario generations can use different
        metadata field names and structures.

        The complete original JSON is preserved under:

            raw_data

        Canonical fields are added without deleting or
        replacing original fields.
        """

        # --------------------------------------------------------------
        # Start with complete original metadata.
        # --------------------------------------------------------------

        metadata = dict(
            data
        )

        # --------------------------------------------------------------
        # Preserve original metadata completely.
        # --------------------------------------------------------------

        metadata["raw_data"] = dict(
            data
        )

        # --------------------------------------------------------------
        # Runtime paths.
        # --------------------------------------------------------------

        metadata["metadata_path"] = str(
            metadata_path
        )

        metadata["scenario_path"] = str(
            metadata_path.parent
        )

        # --------------------------------------------------------------
        # Scenario ID.
        # --------------------------------------------------------------

        metadata["scenario_id"] = (
            self._extract_scenario_id(
                data,
                metadata_path,
            )
        )

        # --------------------------------------------------------------
        # Scenario name.
        # --------------------------------------------------------------

        metadata["name"] = (
            self._extract_first_value(
                data,
                (
                    "name",
                    "scenario_name",
                    "title",
                ),
            )
        )

        # --------------------------------------------------------------
        # Category.
        # --------------------------------------------------------------

        metadata["category"] = (
            self._extract_category(
                data
            )
        )

        # --------------------------------------------------------------
        # Difficulty.
        # --------------------------------------------------------------

        metadata["difficulty"] = (
            self._extract_first_value(
                data,
                (
                    "difficulty",
                    "level",
                ),
            )
        )

        # --------------------------------------------------------------
        # Scenario type.
        # --------------------------------------------------------------

        metadata["scenario_type"] = (
            self._extract_first_value(
                data,
                (
                    "scenario_type",
                    "type",
                ),
            )
        )

        # --------------------------------------------------------------
        # Attack type.
        # --------------------------------------------------------------

        metadata["attack_type"] = (
            self._extract_first_value(
                data,
                (
                    "attack_type",
                    "attack",
                    "attack_category",
                ),
            )
        )

        # --------------------------------------------------------------
        # Tags.
        # --------------------------------------------------------------

        metadata["tags"] = (
            self._extract_tags(
                data
            )
        )

        # --------------------------------------------------------------
        # Vulnerabilities.
        # --------------------------------------------------------------

        metadata["vulnerabilities"] = (
            self._extract_vulnerabilities(
                data
            )
        )

        # --------------------------------------------------------------
        # Vulnerability count.
        # --------------------------------------------------------------

        vulnerability_count = (
            self._extract_first_value(
                data,
                (
                    "vulnerability_count",
                    "num_vulnerabilities",
                ),
            )
        )

        if vulnerability_count is None:

            vulnerability_count = len(
                metadata["vulnerabilities"]
            )

        metadata["vulnerability_count"] = (
            vulnerability_count
        )

        # --------------------------------------------------------------
        # Vulnerability location.
        #
        # Supports BOTH:
        #
        # 1. Object:
        #
        #    "vulnerability_location": {
        #        "file": "...",
        #        "function": "..."
        #    }
        #
        # 2. List:
        #
        #    "vulnerability_location": [
        #        {
        #            "file": "...",
        #            "function": "..."
        #        }
        #    ]
        #
        # The original value is still preserved inside raw_data.
        # --------------------------------------------------------------

        metadata["vulnerability_location"] = (
            self._extract_vulnerability_location(
                data
            )
        )

        # --------------------------------------------------------------
        # Valid attacks.
        # --------------------------------------------------------------

        metadata["valid_attacks"] = (
            self._extract_valid_attacks(
                data
            )
        )

        # --------------------------------------------------------------
        # Valid defenses.
        # --------------------------------------------------------------

        metadata["valid_defenses"] = (
            self._extract_valid_defenses(
                data
            )
        )

        # --------------------------------------------------------------
        # Optimal attack.
        # --------------------------------------------------------------

        metadata["optimal_attack"] = (
            self._extract_first_value(
                data,
                (
                    "optimal_attack",
                    "best_attack",
                ),
            )
        )

        # --------------------------------------------------------------
        # Optimal defense.
        # --------------------------------------------------------------

        metadata["optimal_defense"] = (
            self._extract_first_value(
                data,
                (
                    "optimal_defense",
                    "best_defense",
                ),
            )
        )

        # --------------------------------------------------------------
        # Root cause.
        # --------------------------------------------------------------

        metadata["root_cause"] = (
            self._extract_root_cause(
                data
            )
        )

        # --------------------------------------------------------------
        # Relevant files.
        # --------------------------------------------------------------

        metadata["relevant_files"] = (
            self._extract_relevant_files(
                data
            )
        )

        # --------------------------------------------------------------
        # Noise files.
        # --------------------------------------------------------------

        metadata["noise_files"] = (
            self._extract_noise_files(
                data
            )
        )

        # --------------------------------------------------------------
        # Vulnerability interactions.
        # --------------------------------------------------------------

        metadata["vulnerability_interactions"] = (
            self._extract_vulnerability_interactions(
                data
            )
        )

        # --------------------------------------------------------------
        # Source.
        # --------------------------------------------------------------

        metadata["source"] = (
            self._extract_first_value(
                data,
                (
                    "source",
                    "source_file",
                ),
            )
        )

        # --------------------------------------------------------------
        # Sink.
        # --------------------------------------------------------------

        metadata["sink"] = (
            self._extract_first_value(
                data,
                (
                    "sink",
                    "sink_file",
                ),
            )
        )

        # --------------------------------------------------------------
        # Entry point.
        # --------------------------------------------------------------

        metadata["entry_point"] = (
            self._extract_first_value(
                data,
                (
                    "entry_point",
                    "entry",
                ),
            )
        )

        # --------------------------------------------------------------
        # Injection point.
        # --------------------------------------------------------------

        metadata["injection_point"] = (
            self._extract_first_value(
                data,
                (
                    "injection_point",
                ),
            )
        )

        # --------------------------------------------------------------
        # CWE reference.
        # --------------------------------------------------------------

        metadata["cwe_reference"] = (
            self._extract_first_value(
                data,
                (
                    "cwe_reference",
                    "cwe",
                ),
            )
        )

        # --------------------------------------------------------------
        # Cross-file reasoning.
        # --------------------------------------------------------------

        metadata["requires_cross_file_reasoning"] = (
            self._extract_first_value(
                data,
                (
                    "requires_cross_file_reasoning",
                ),
            )
        )

        # --------------------------------------------------------------
        # Reasoning depth.
        # --------------------------------------------------------------

        metadata["reasoning_depth"] = (
            self._extract_first_value(
                data,
                (
                    "reasoning_depth",
                ),
            )
        )

        # --------------------------------------------------------------
        # Validate normalized representation.
        # --------------------------------------------------------------

        self._validate_normalized_metadata(
            metadata
        )

        return metadata

    # ==================================================================
    # SCENARIO ID
    # ==================================================================

    @staticmethod
    def _extract_scenario_id(
        data: Dict[str, Any],
        metadata_path: Path,
    ) -> Optional[str]:
        """
        Extract scenario ID.

        Supports:

            scenario_id
            id

        Falls back to the scenario directory name.
        """

        value = data.get(
            "scenario_id"
        )

        if value is None:

            value = data.get(
                "id"
            )

        if value is not None:

            return str(
                value
            )

        parent_name = (
            metadata_path.parent.name
        )

        if parent_name:

            return parent_name

        return None

    # ==================================================================
    # CATEGORY
    # ==================================================================

    @staticmethod
    def _extract_category(
        data: Dict[str, Any],
    ) -> Optional[str]:
        """
        Extract scenario category.
        """

        for key in (
            "category",
            "domain",
            "scenario_category",
            "attack_category",
        ):

            value = data.get(
                key
            )

            if value is not None:

                return str(
                    value
                )

        return None

    # ==================================================================
    # TAGS
    # ==================================================================

    @staticmethod
    def _extract_tags(
        data: Dict[str, Any],
    ) -> List[str]:
        """
        Extract scenario tags.
        """

        tags = data.get(
            "tags",
            []
        )

        if isinstance(
            tags,
            str,
        ):

            return [
                tags
            ] if tags.strip() else []

        if not isinstance(
            tags,
            list,
        ):

            return []

        return MetadataLoader._unique_strings(
            MetadataLoader._clean_string_list(
                tags
            )
        )

    # ==================================================================
    # VULNERABILITIES
    # ==================================================================

    @staticmethod
    def _extract_vulnerabilities(
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract vulnerability definitions.

        Unknown fields are preserved.
        """

        vulnerabilities = data.get(
            "vulnerabilities",
            []
        )

        if vulnerabilities is None:

            return []

        if not isinstance(
            vulnerabilities,
            list,
        ):

            return []

        result = []

        for vulnerability in vulnerabilities:

            if not isinstance(
                vulnerability,
                dict,
            ):

                continue

            result.append(
                dict(
                    vulnerability
                )
            )

        return result

    # ==================================================================
    # VULNERABILITY LOCATION
    # ==================================================================

    @staticmethod
    def _extract_vulnerability_location(
        data: Dict[str, Any],
    ) -> Any:
        """
        Extract vulnerability_location.

        Supported forms:

            {
                "file": "...",
                "function": "..."
            }

        OR:

            [
                {
                    "file": "...",
                    "function": "..."
                },
                ...
            ]

        The structure is deliberately preserved rather than
        forcing one representation, because existing scenarios
        use both forms.

        Returns
        -------
        dict | list | None
        """

        location = data.get(
            "vulnerability_location"
        )

        if location is None:

            return None

        # --------------------------------------------------------------
        # Single vulnerability location object.
        # --------------------------------------------------------------

        if isinstance(
            location,
            dict,
        ):

            return dict(
                location
            )

        # --------------------------------------------------------------
        # Multiple vulnerability locations.
        # --------------------------------------------------------------

        if isinstance(
            location,
            list,
        ):

            result = []

            for item in location:

                if isinstance(
                    item,
                    dict,
                ):

                    result.append(
                        dict(
                            item
                        )

                    )

                else:

                    # Preserve unusual but valid scalar
                    # location entries as-is.
                    result.append(
                        item
                    )

            return result

        # --------------------------------------------------------------
        # Unknown scalar form.
        #
        # Do not throw away information.
        # --------------------------------------------------------------

        return location

    # ==================================================================
    # VALID ATTACKS
    # ==================================================================

    @classmethod
    def _extract_valid_attacks(
        cls,
        data: Dict[str, Any],
    ) -> List[str]:
        """
        Extract valid attacks.

        Primary source:

            valid_attacks

        Fallbacks:

            vulnerability_interactions[].label
            optimal_attack
            best_attack
        """

        attacks = data.get(
            "valid_attacks"
        )

        if isinstance(
            attacks,
            str,
        ):

            return (
                [
                    attacks.strip()
                ]
                if attacks.strip()
                else []
            )

        if isinstance(
            attacks,
            list,
        ):

            return cls._unique_strings(
                cls._clean_string_list(
                    attacks
                )
            )

        result: List[str] = []

        # --------------------------------------------------------------
        # Interaction labels.
        # --------------------------------------------------------------

        interactions = data.get(
            "vulnerability_interactions",
            []
        )

        if isinstance(
            interactions,
            list,
        ):

            for interaction in interactions:

                if not isinstance(
                    interaction,
                    dict,
                ):

                    continue

                label = interaction.get(
                    "label"
                )

                if label is not None:

                    result.append(
                        str(
                            label
                        )
                    )

        # --------------------------------------------------------------
        # Optimal attack.
        # --------------------------------------------------------------

        optimal_attack = cls._extract_first_value(
            data,
            (
                "optimal_attack",
                "best_attack",
            ),
        )

        if optimal_attack is not None:

            if isinstance(
                optimal_attack,
                list,
            ):

                result.extend(
                    cls._clean_string_list(
                        optimal_attack
                    )
                )

            else:

                result.append(
                    str(
                        optimal_attack
                    )
                )

        return cls._unique_strings(
            cls._clean_string_list(
                result
            )
        )

    # ==================================================================
    # VALID DEFENSES
    # ==================================================================

    @classmethod
    def _extract_valid_defenses(
        cls,
        data: Dict[str, Any],
    ) -> List[str]:
        """
        Extract valid defenses.

        Primary source:

            valid_defenses

        Fallback:

            optimal_defense
            best_defense
        """

        defenses = data.get(
            "valid_defenses"
        )

        if isinstance(
            defenses,
            str,
        ):

            return (
                [
                    defenses.strip()
                ]
                if defenses.strip()
                else []
            )

        if isinstance(
            defenses,
            list,
        ):

            return cls._unique_strings(
                cls._clean_string_list(
                    defenses
                )
            )

        result: List[str] = []

        optimal_defense = cls._extract_first_value(
            data,
            (
                "optimal_defense",
                "best_defense",
            ),
        )

        if optimal_defense is not None:

            if isinstance(
                optimal_defense,
                list,
            ):

                result.extend(
                    cls._clean_string_list(
                        optimal_defense
                    )
                )

            else:

                result.append(
                    str(
                        optimal_defense
                    )
                )

        return cls._unique_strings(
            cls._clean_string_list(
                result
            )
        )

    # ==================================================================
    # ROOT CAUSE
    # ==================================================================

    @staticmethod
    def _extract_root_cause(
        data: Dict[str, Any],
    ) -> Any:
        """
        Extract root cause.

        Supports string, list, dictionary, or other
        structured representations.
        """

        if "root_cause" in data:

            return data[
                "root_cause"
            ]

        if "root_causes" in data:

            return data[
                "root_causes"
            ]

        return None

    # ==================================================================
    # RELEVANT FILES
    # ==================================================================

    @classmethod
    def _extract_relevant_files(
        cls,
        data: Dict[str, Any],
    ) -> List[str]:
        """
        Extract scenario-level and vulnerability-level
        relevant files.
        """

        result: List[str] = []

        # --------------------------------------------------------------
        # Scenario-level relevant files.
        # --------------------------------------------------------------

        direct = data.get(
            "relevant_files",
            []
        )

        if isinstance(
            direct,
            str,
        ):

            result.append(
                direct
            )

        elif isinstance(
            direct,
            list,
        ):

            result.extend(
                str(item)
                for item in direct
                if item is not None
            )

        # --------------------------------------------------------------
        # Vulnerability-level relevant files.
        # --------------------------------------------------------------

        vulnerabilities = data.get(
            "vulnerabilities",
            []
        )

        if isinstance(
            vulnerabilities,
            list,
        ):

            for vulnerability in vulnerabilities:

                if not isinstance(
                    vulnerability,
                    dict,
                ):

                    continue

                files = vulnerability.get(
                    "relevant_files",
                    []
                )

                if isinstance(
                    files,
                    str,
                ):

                    result.append(
                        files
                    )

                elif isinstance(
                    files,
                    list,
                ):

                    result.extend(
                        str(item)
                        for item in files
                        if item is not None
                    )

        return cls._unique_paths(
            result
        )

    # ==================================================================
    # NOISE FILES
    # ==================================================================

    @classmethod
    def _extract_noise_files(
        cls,
        data: Dict[str, Any],
    ) -> List[str]:
        """
        Extract explicitly declared noise files.
        """

        noise = data.get(
            "noise_files",
            []
        )

        if isinstance(
            noise,
            str,
        ):

            noise = [
                noise
            ]

        if not isinstance(
            noise,
            list,
        ):

            return []

        return cls._unique_paths(
            [
                str(item)
                for item in noise
                if item is not None
            ]
        )

    # ==================================================================
    # VULNERABILITY INTERACTIONS
    # ==================================================================

    @staticmethod
    def _extract_vulnerability_interactions(
        data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract vulnerability interaction definitions.

        Interaction objects are preserved.
        """

        interactions = data.get(
            "vulnerability_interactions",
            []
        )

        if interactions is None:

            return []

        if not isinstance(
            interactions,
            list,
        ):

            return []

        result = []

        for interaction in interactions:

            if not isinstance(
                interaction,
                dict,
            ):

                continue

            result.append(
                dict(
                    interaction
                )
            )

        return result

    # ==================================================================
    # GENERIC FIRST VALUE
    # ==================================================================

    @staticmethod
    def _extract_first_value(
        data: Dict[str, Any],
        keys: Sequence[str],
    ) -> Any:
        """
        Return first existing non-null value.
        """

        for key in keys:

            if key not in data:

                continue

            value = data[
                key
            ]

            if value is not None:

                return value

        return None

    # ==================================================================
    # CLEAN STRING LIST
    # ==================================================================

    @staticmethod
    def _clean_string_list(
        values: Sequence[Any],
    ) -> List[str]:
        """
        Convert values into clean non-empty strings.
        """

        result = []

        for value in values:

            if value is None:

                continue

            value = str(
                value
            ).strip()

            if not value:

                continue

            result.append(
                value
            )

        return result

    # ==================================================================
    # UNIQUE STRINGS
    # ==================================================================

    @staticmethod
    def _unique_strings(
        values: Sequence[str],
    ) -> List[str]:
        """
        Deduplicate strings while preserving order.
        """

        result = []
        seen = set()

        for value in values:

            normalized = str(
                value
            ).strip()

            if not normalized:

                continue

            key = normalized.casefold()

            if key in seen:

                continue

            seen.add(
                key
            )

            result.append(
                normalized
            )

        return result

    # ==================================================================
    # UNIQUE PATHS
    # ==================================================================

    @staticmethod
    def _unique_paths(
        values: Sequence[str],
    ) -> List[str]:
        """
        Deduplicate paths while preserving original spelling.

        Comparison is case-insensitive and slash-insensitive.
        """

        result = []
        seen = set()

        for value in values:

            value = str(
                value
            ).strip()

            if not value:

                continue

            normalized = (
                value
                .replace(
                    "\\",
                    "/",
                )
                .casefold()
                .strip()
            )

            if normalized in seen:

                continue

            seen.add(
                normalized
            )

            result.append(
                value
            )

        return result

    # ==================================================================
    # NORMALIZED METADATA VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_normalized_metadata(
        metadata: Dict[str, Any],
    ) -> None:
        """
        Validate the normalized representation.

        This deliberately does NOT require every scenario to
        physically contain every canonical field.

        Different scenario generations use different schemas.

        Missing optional semantic fields are represented as:

            None
            []

        rather than causing the entire scenario to fail.
        """

        if not isinstance(
            metadata,
            dict,
        ):

            raise ValueError(
                "Normalized metadata must be a dictionary."
            )

        # --------------------------------------------------------------
        # Scenario ID must be recoverable.
        # --------------------------------------------------------------

        if not metadata.get(
            "scenario_id"
        ):

            raise ValueError(
                "metadata.json does not contain a usable "
                "scenario identifier."
            )

        # --------------------------------------------------------------
        # Canonical collection fields.
        # --------------------------------------------------------------

        collection_fields = [
            "vulnerabilities",
            "valid_attacks",
            "valid_defenses",
            "relevant_files",
            "noise_files",
            "tags",
            "vulnerability_interactions",
        ]

        for field in collection_fields:

            value = metadata.get(
                field
            )

            if not isinstance(
                value,
                list,
            ):

                raise ValueError(
                    "Normalized metadata field "
                    f"'{field}' must be a list."
                )

        # --------------------------------------------------------------
        # Vulnerability objects.
        # --------------------------------------------------------------

        for vulnerability in metadata[
            "vulnerabilities"
        ]:

            if not isinstance(
                vulnerability,
                dict,
            ):

                raise ValueError(
                    "Every item in normalized "
                    "'vulnerabilities' must be an object."
                )

        # --------------------------------------------------------------
        # Vulnerability location.
        #
        # Both dict and list forms are valid.
        # --------------------------------------------------------------

        location = metadata.get(
            "vulnerability_location"
        )

        if location is not None:

            if not isinstance(
                location,
                (
                    dict,
                    list,
                    str,
                ),
            ):

                raise ValueError(
                    "Normalized metadata field "
                    "'vulnerability_location' must be "
                    "an object, list, string, or null."
                )

    # ==================================================================
    # PUBLIC HELPERS
    # ==================================================================

    @staticmethod
    def get_vulnerabilities(
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Return vulnerability definitions.
        """

        return metadata.get(
            "vulnerabilities",
            []
        )

    @staticmethod
    def get_valid_attacks(
        metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Return attacks considered valid.
        """

        return metadata.get(
            "valid_attacks",
            []
        )

    @staticmethod
    def get_valid_defenses(
        metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Return defenses considered valid.
        """

        return metadata.get(
            "valid_defenses",
            []
        )

    @staticmethod
    def get_relevant_files(
        metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Return relevant files.
        """

        return metadata.get(
            "relevant_files",
            []
        )

    @staticmethod
    def get_noise_files(
        metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Return noise files.
        """

        return metadata.get(
            "noise_files",
            []
        )

    @staticmethod
    def get_optimal_attack(
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """
        Return optimal attack.
        """

        return metadata.get(
            "optimal_attack"
        )

    @staticmethod
    def get_optimal_defense(
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """
        Return optimal defense.
        """

        return metadata.get(
            "optimal_defense"
        )

    @staticmethod
    def get_category(
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """
        Return scenario category.
        """

        return metadata.get(
            "category"
        )

    @staticmethod
    def get_attack_type(
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """
        Return primary attack type.
        """

        return metadata.get(
            "attack_type"
        )

    @staticmethod
    def get_difficulty(
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """
        Return scenario difficulty.
        """

        return metadata.get(
            "difficulty"
        )

    @staticmethod
    def get_scenario_id(
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """
        Return scenario ID.
        """

        return metadata.get(
            "scenario_id"
        )

    @staticmethod
    def get_tags(
        metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Return scenario tags.
        """

        return metadata.get(
            "tags",
            []
        )

    # ==================================================================
    # VULNERABILITY LOCATION HELPER
    # ==================================================================

    @staticmethod
    def get_vulnerability_location(
        metadata: Dict[str, Any],
    ) -> Any:
        """
        Return vulnerability location.

        Can be either:

            dict

        or:

            list[dict]

        depending on the original scenario schema.
        """

        return metadata.get(
            "vulnerability_location"
        )

    # ==================================================================
    # VULNERABILITY HELPERS
    # ==================================================================

    @staticmethod
    def get_vulnerability_ids(
        metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Return vulnerability IDs.
        """

        vulnerabilities = metadata.get(
            "vulnerabilities",
            []
        )

        return [
            str(
                vulnerability["id"]
            )
            for vulnerability in vulnerabilities
            if isinstance(
                vulnerability,
                dict,
            )
            and "id" in vulnerability
        ]

    @staticmethod
    def get_vulnerability_types(
        metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Return vulnerability types.
        """

        vulnerabilities = metadata.get(
            "vulnerabilities",
            []
        )

        return [
            str(
                vulnerability["type"]
            )
            for vulnerability in vulnerabilities
            if isinstance(
                vulnerability,
                dict,
            )
            and "type" in vulnerability
        ]

    @staticmethod
    def get_vulnerability(
        metadata: Dict[str, Any],
        vulnerability_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Find vulnerability by ID.
        """

        vulnerabilities = metadata.get(
            "vulnerabilities",
            []
        )

        for vulnerability in vulnerabilities:

            if not isinstance(
                vulnerability,
                dict,
            ):

                continue

            if str(
                vulnerability.get("id")
            ) == str(
                vulnerability_id
            ):

                return vulnerability

        return None

    # ==================================================================
    # INTERACTION HELPERS
    # ==================================================================

    @staticmethod
    def get_interactions_for_vulnerability(
        metadata: Dict[str, Any],
        vulnerability_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Return interactions involving a vulnerability.
        """

        interactions = metadata.get(
            "vulnerability_interactions",
            []
        )

        result = []

        for interaction in interactions:

            if not isinstance(
                interaction,
                dict,
            ):

                continue

            vulnerability_ids = (
                interaction.get(
                    "vulnerabilities",
                    []
                )
            )

            if isinstance(
                vulnerability_ids,
                str,
            ):

                vulnerability_ids = [
                    vulnerability_ids
                ]

            if (
                isinstance(
                    vulnerability_ids,
                    list,
                )
                and str(
                    vulnerability_id
                ) in {
                    str(item)
                    for item in vulnerability_ids
                }
            ):

                result.append(
                    interaction
                )

        return result

    # ==================================================================
    # SUMMARY
    # ==================================================================

    @staticmethod
    def get_summary(
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return compact metadata summary.
        """

        vulnerabilities = metadata.get(
            "vulnerabilities",
            []
        )

        location = metadata.get(
            "vulnerability_location"
        )

        if isinstance(
            location,
            list,
        ):

            vulnerability_location_count = len(
                location
            )

        elif location is not None:

            vulnerability_location_count = 1

        else:

            vulnerability_location_count = 0

        return {
            "scenario_id": metadata.get(
                "scenario_id"
            ),
            "name": metadata.get(
                "name"
            ),
            "difficulty": metadata.get(
                "difficulty"
            ),
            "category": metadata.get(
                "category"
            ),
            "attack_type": metadata.get(
                "attack_type"
            ),
            "num_vulnerabilities": len(
                vulnerabilities
            ),
            "num_vulnerability_locations": (
                vulnerability_location_count
            ),
            "num_valid_attacks": len(
                metadata.get(
                    "valid_attacks",
                    []
                )
            ),
            "num_valid_defenses": len(
                metadata.get(
                    "valid_defenses",
                    []
                )
            ),
            "num_relevant_files": len(
                metadata.get(
                    "relevant_files",
                    []
                )
            ),
            "num_vulnerability_interactions": len(
                metadata.get(
                    "vulnerability_interactions",
                    []
                )
            ),
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def load_metadata(
    metadata_path: Path | str,
) -> Dict[str, Any]:
    """
    Convenience wrapper around MetadataLoader.
    """

    loader = MetadataLoader()

    return loader.load(
        metadata_path
    )