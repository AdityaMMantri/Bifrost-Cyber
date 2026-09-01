"""
scenario_loader.py

Loads scenario-level information for SOGARL.

Responsibilities
----------------
- Discover scenario directories
- Validate scenario structure
- Load metadata.json through MetadataLoader
- Load scenario.md
- Parse both Markdown-heading and legacy "Field:" formats
- Keep only configured safe scenario.md sections
- Create Scenario objects

Does NOT:
- Load source-code contents
- Select relevant/noise files
- Build prompts
- Generate model responses
- Evaluate responses
- Calculate rewards
- Perform GRPO
- Load red_sft.json / blue_sft.json
- Expose metadata.json to the model
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from configs.config import SAFE_SCENARIO_SECTIONS

from src.data.models import Scenario
from src.data.metadata_loader import MetadataLoader


# ============================================================================
# CONSTANTS
# ============================================================================

SCENARIO_PATTERN = re.compile(
    r"^scenario_(\d+)$",
    re.IGNORECASE,
)

METADATA_FILENAME = "metadata.json"
SCENARIO_FILENAME = "scenario.md"
CODEBASE_DIRECTORY = "files"


# ============================================================================
# SCENARIO LOADER
# ============================================================================

class ScenarioLoader:
    """
    Discovers and loads SOGARL scenarios.

    A valid scenario contains:

        scenario.md
        metadata.json
        files/

    This class loads scenario-level information only.

    Source-code contents are loaded later by:

        PathResolver
            ↓
        CodeLoader

    IMPORTANT
    ---------
    scenario.md is treated as potentially sensitive.

    Only explicitly approved sections from
    SAFE_SCENARIO_SECTIONS are returned as
    Scenario.scenario_description.

    metadata.json is never included in the
    model-visible scenario description.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        metadata_loader: Optional[MetadataLoader] = None,
        logger=None,
        strict: bool = True,
    ) -> None:

        self.dataset_path = (
            Path(dataset_path)
            .expanduser()
            .resolve()
        )

        self.metadata_loader = (
            metadata_loader
            if metadata_loader is not None
            else MetadataLoader(logger=logger)
        )

        self.logger = logger
        self.strict = strict

        self._validate_dataset_path()

    # ========================================================================
    # DATASET
    # ========================================================================

    def _validate_dataset_path(self) -> None:
        """
        Validate the SFT_Dataset root.
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist:\n"
                f"{self.dataset_path}"
            )

        if not self.dataset_path.is_dir():
            raise NotADirectoryError(
                f"Dataset path is not a directory:\n"
                f"{self.dataset_path}"
            )

    # ========================================================================
    # DISCOVERY
    # ========================================================================

    def discover_scenarios(
        self,
        sort: bool = True,
    ) -> List[Path]:
        """
        Discover scenario directories.

        Only directories matching:

            scenario_<number>

        are returned.
        """

        scenarios = [
            path
            for path in self.dataset_path.iterdir()
            if path.is_dir()
            and SCENARIO_PATTERN.match(path.name)
        ]

        if sort:
            scenarios.sort(
                key=self._scenario_sort_key
            )

        return scenarios

    def discover_scenario_ids(
        self,
        sort: bool = True,
    ) -> List[str]:
        """
        Return discovered scenario IDs.
        """

        return [
            path.name
            for path in self.discover_scenarios(sort)
        ]

    def count_scenarios(self) -> int:
        """
        Return number of discovered scenarios.
        """

        return len(
            self.discover_scenarios()
        )

    # ========================================================================
    # LOAD ONE SCENARIO
    # ========================================================================

    def load_scenario(
        self,
        scenario_id: str | int,
    ) -> Scenario:
        """
        Load one scenario.

        The public API is intentionally unchanged.
        """

        scenario_name = (
            self._normalize_scenario_id(
                scenario_id
            )
        )

        scenario_path = (
            self.dataset_path / scenario_name
        )

        validation = self.validate_scenario(
            scenario_path
        )

        if not validation["valid"]:

            if self.strict:

                problems = "\n".join(
                    validation["errors"]
                )

                raise ValueError(
                    f"Invalid scenario "
                    f"{scenario_name}:\n"
                    f"{problems}"
                )

        metadata_path = (
            scenario_path / METADATA_FILENAME
        )

        scenario_md_path = (
            scenario_path / SCENARIO_FILENAME
        )

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        metadata = self.metadata_loader.load(
            metadata_path
        )

        # --------------------------------------------------------------
        # Safe scenario description
        # --------------------------------------------------------------

        scenario_description = (
            self._load_safe_scenario_sections(
                scenario_md_path
            )
        )

        # --------------------------------------------------------------
        # Scenario object
        # --------------------------------------------------------------

        scenario = Scenario(
            scenario_id=scenario_name,
            scenario_path=scenario_path,
            scenario_description=scenario_description,
            metadata=metadata,
        )

        self._log_loaded_scenario(
            scenario
        )

        return scenario

    # ========================================================================
    # LOAD MANY SCENARIOS
    # ========================================================================

    def load_scenarios(
        self,
        scenario_ids: Optional[
            Sequence[str | int]
        ] = None,
    ) -> List[Scenario]:
        """
        Load multiple scenarios.

        If scenario_ids is None, every discovered scenario
        is loaded in numerical order.
        """

        if scenario_ids is None:
            scenario_ids = (
                self.discover_scenario_ids()
            )

        scenarios = []

        for scenario_id in scenario_ids:

            scenarios.append(
                self.load_scenario(
                    scenario_id
                )
            )

        return scenarios

    # ========================================================================
    # SCENARIO.MD
    # ========================================================================

    def _load_safe_scenario_sections(
        self,
        scenario_md_path: Path,
    ) -> str:
        """
        Read scenario.md and retain ONLY approved sections.

        Supported formats
        ------------------

        Markdown headings:

            # Description
            ## Description
            ### Architecture

        Legacy fields:

            Description:
            Overview:
            Technology Stack:

        Mixed documents are supported.

        IMPORTANT
        ---------
        Sensitive sections such as:

            Expected Attack
            Expected Defense
            Primary Vulnerability
            Success Condition
            Attack
            Vulnerability

        are NOT included unless explicitly present in
        SAFE_SCENARIO_SECTIONS.

        Returns
        -------
        str
            Sanitized scenario description.
        """

        if not scenario_md_path.exists():

            if self.strict:
                raise FileNotFoundError(
                    f"scenario.md not found:\n"
                    f"{scenario_md_path}"
                )

            return ""

        if not scenario_md_path.is_file():

            raise ValueError(
                f"scenario.md is not a file:\n"
                f"{scenario_md_path}"
            )

        try:

            markdown = scenario_md_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            markdown = scenario_md_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        return self._extract_safe_sections(
            markdown
        )

    # ========================================================================
    # SECTION EXTRACTION
    # ========================================================================

    def _extract_safe_sections(
        self,
        markdown: str,
    ) -> str:
        """
        Parse scenario.md and extract safe sections.

        Handles:

            1. Markdown headings
            2. Legacy "Field:" headings
            3. Mixed Markdown + legacy formats
            4. Fenced code blocks
            5. Nested headings

        The parser first creates a structural representation
        of the document.

        Only after parsing are safe sections selected.

        This prevents the old problem where a legacy field such
        as "Description:" was never recognized and therefore
        produced:

            Safe scenario text: 0 characters
        """

        if not markdown or not markdown.strip():
            return ""

        sections = (
            self._parse_document_sections(
                markdown
            )
        )

        safe_sections = []

        for section in sections:

            title = self._normalize_section_title(
                section["title"]
            )

            if not self._is_safe_section(
                title
            ):
                continue

            content = section["content"].strip()

            if not content:
                continue

            # Keep the section title so the model receives
            # structured context.
            safe_sections.append(
                f"{title}:\n{content}"
            )

        return self._clean_sections(
            safe_sections
        )

    # ========================================================================
    # GENERIC DOCUMENT PARSER
    # ========================================================================

    def _parse_document_sections(
        self,
        markdown: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse scenario.md into sections.

        Supports both:

            ## Description

        and:

            Description:

        Markdown headings are assigned their actual level.

        Legacy fields are treated as level 1 structural
        sections.

        Fenced code blocks are protected from heading detection.
        """

        lines = markdown.splitlines()

        sections: List[
            Dict[str, Any]
        ] = []

        current_title: Optional[str] = None
        current_level: Optional[int] = None
        current_lines: List[str] = []

        inside_code_block = False
        fence_marker: Optional[str] = None

        def flush() -> None:

            nonlocal current_title
            nonlocal current_level
            nonlocal current_lines

            if current_title is None:
                return

            content = "\n".join(
                current_lines
            ).strip()

            sections.append(
                {
                    "title": current_title,
                    "level": current_level or 1,
                    "content": content,
                }
            )

            current_title = None
            current_level = None
            current_lines = []

        for line in lines:

            stripped = line.strip()

            # ----------------------------------------------------------
            # Fenced code block handling
            # ----------------------------------------------------------

            fence_match = re.match(
                r"^(```+|~~~+)",
                stripped,
            )

            if fence_match:

                marker = fence_match.group(1)

                if not inside_code_block:

                    inside_code_block = True
                    fence_marker = marker[0]

                else:

                    if (
                        fence_marker is None
                        or marker.startswith(
                            fence_marker
                        )
                    ):
                        inside_code_block = False
                        fence_marker = None

                # Code belongs to current section.
                if current_title is not None:
                    current_lines.append(line)

                continue

            # ----------------------------------------------------------
            # Everything inside a code block is content.
            # ----------------------------------------------------------

            if inside_code_block:

                if current_title is not None:
                    current_lines.append(line)

                continue

            # ----------------------------------------------------------
            # Markdown heading
            # ----------------------------------------------------------

            heading_match = re.match(
                r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)\s*$",
                line,
            )

            if heading_match:

                level = len(
                    heading_match.group(1)
                )

                title = (
                    heading_match.group(2)
                    .strip()
                )

                # Remove optional trailing Markdown
                # heading syntax.
                title = title.rstrip("#").strip()

                flush()

                current_title = title
                current_level = level
                current_lines = []

                continue

            # ----------------------------------------------------------
            # Legacy "Field:" format
            # ----------------------------------------------------------

            legacy_title = (
                self._detect_legacy_heading(
                    line
                )
            )

            if legacy_title is not None:

                flush()

                current_title = legacy_title
                current_level = 1
                current_lines = []

                continue

            # ----------------------------------------------------------
            # Normal content
            # ----------------------------------------------------------

            if current_title is not None:

                current_lines.append(line)

        flush()

        return sections

    # ========================================================================
    # LEGACY HEADING DETECTION
    # ========================================================================

    @staticmethod
    def _detect_legacy_heading(
        line: str,
    ) -> Optional[str]:
        """
        Detect legacy fields such as:

            Description:
            Technology Stack:
            Expected Attack:

        This intentionally uses conservative rules so ordinary
        prose containing a colon is not accidentally interpreted
        as a section heading.
        """

        stripped = line.strip()

        if not stripped:
            return None

        if stripped.startswith(
            ("-", "*", ">", "|")
        ):
            return None

        if stripped.startswith(
            ("```", "~~~")
        ):
            return None

        if not stripped.endswith(":"):
            return None

        # Avoid treating long prose sentences as headings.
        if len(stripped) > 100:
            return None

        title = stripped[:-1].strip()

        if not title:
            return None

        # A legacy heading should look like a field title,
        # not a complete sentence.
        if title[-1:] in {
            ".",
            "?",
            "!",
            ";",
        }:
            return None

        return title

    # ========================================================================
    # SAFE SECTION CHECK
    # ========================================================================

    @classmethod
    def _is_safe_section(
        cls,
        title: str,
    ) -> bool:
        """
        Determine whether a section is allowed to reach
        the model.

        Matching is case-insensitive and whitespace-normalized.

        SAFE_SCENARIO_SECTIONS remains the single source
        of truth from configs/config.py.
        """

        normalized = (
            cls._normalize_section_title(
                title
            )
        )

        safe_lookup = {
            cls._normalize_section_title(
                item
            ).casefold()
            for item in SAFE_SCENARIO_SECTIONS
        }

        return (
            normalized.casefold()
            in safe_lookup
        )

    # ========================================================================
    # SECTION TITLE NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_section_title(
        title: str,
    ) -> str:
        """
        Normalize section titles.

        Examples:

            "Description:"        → "Description"
            " Description "       → "Description"
            "Technology   Stack"  → "Technology Stack"
            "Overview---"         → "Overview"
        """

        title = str(
            title
        ).strip()

        # Remove trailing colon.
        title = re.sub(
            r":+$",
            "",
            title,
        )

        # Remove trailing dash decoration only.
        title = re.sub(
            r"\s+-+\s*$",
            "",
            title,
        )

        # Collapse whitespace.
        title = re.sub(
            r"\s+",
            " ",
            title,
        )

        return title.strip()

    # ========================================================================
    # CLEAN
    # ========================================================================

    @staticmethod
    def _clean_sections(
        sections: List[str],
    ) -> str:
        """
        Remove empty sections and join safe sections.
        """

        cleaned = []

        for section in sections:

            section = section.strip()

            if section:
                cleaned.append(
                    section
                )

        return "\n\n".join(
            cleaned
        )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def validate_scenario(
        self,
        scenario_path: Path | str,
    ) -> dict:
        """
        Validate scenario structure.

        Required:

            scenario directory
            metadata.json
            scenario.md
            files/

        Metadata contents are validated by MetadataLoader
        when the scenario is actually loaded.
        """

        scenario_path = (
            Path(scenario_path)
            .expanduser()
            .resolve()
        )

        errors = []
        warnings = []

        # --------------------------------------------------------------
        # Scenario directory
        # --------------------------------------------------------------

        if not scenario_path.exists():

            errors.append(
                "Scenario directory does not exist."
            )

            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
            }

        if not scenario_path.is_dir():

            errors.append(
                "Scenario path is not a directory."
            )

            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
            }

        # --------------------------------------------------------------
        # Scenario name
        # --------------------------------------------------------------

        if not SCENARIO_PATTERN.match(
            scenario_path.name
        ):

            warnings.append(
                f"Directory name "
                f"'{scenario_path.name}' does not "
                f"follow scenario_<number> format."
            )

        # --------------------------------------------------------------
        # metadata.json
        # --------------------------------------------------------------

        metadata_path = (
            scenario_path
            / METADATA_FILENAME
        )

        if not metadata_path.is_file():

            errors.append(
                "metadata.json is missing."
            )

        # --------------------------------------------------------------
        # scenario.md
        # --------------------------------------------------------------

        scenario_md_path = (
            scenario_path
            / SCENARIO_FILENAME
        )

        if not scenario_md_path.is_file():

            errors.append(
                "scenario.md is missing."
            )

        # --------------------------------------------------------------
        # files/
        # --------------------------------------------------------------

        codebase_path = (
            scenario_path
            / CODEBASE_DIRECTORY
        )

        if not codebase_path.is_dir():

            errors.append(
                "files/ directory is missing."
            )

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    # ========================================================================
    # SORTING
    # ========================================================================

    @staticmethod
    def _scenario_sort_key(
        path: Path,
    ) -> int:
        """
        Sort scenarios numerically.

        Example:

            scenario_002
            scenario_010
            scenario_100
        """

        match = SCENARIO_PATTERN.match(
            path.name
        )

        if match is None:
            return 10**9

        return int(
            match.group(1)
        )

    # ========================================================================
    # ID NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_scenario_id(
        scenario_id: str | int,
    ) -> str:
        """
        Normalize user-provided scenario IDs.

        Accepted:

            1
            "1"
            "001"
            "scenario_1"
            "scenario_001"

        All become:

            scenario_001
        """

        if isinstance(
            scenario_id,
            bool,
        ):
            raise TypeError(
                "Scenario ID must be a string or integer, "
                "not bool."
            )

        if isinstance(
            scenario_id,
            int,
        ):

            if scenario_id < 0:

                raise ValueError(
                    "Scenario ID cannot be negative."
                )

            return (
                f"scenario_{scenario_id:03d}"
            )

        value = str(
            scenario_id
        ).strip()

        if not value:

            raise ValueError(
                "Scenario ID cannot be empty."
            )

        if value.isdigit():

            return (
                f"scenario_{int(value):03d}"
            )

        match = re.match(
            r"^scenario_(\d+)$",
            value,
            re.IGNORECASE,
        )

        if match:

            return (
                f"scenario_{int(match.group(1)):03d}"
            )

        raise ValueError(
            f"Invalid scenario ID: {scenario_id!r}. "
            f"Expected 'scenario_001', '001', or 1."
        )

    # ========================================================================
    # NUMERIC ID
    # ========================================================================

    @staticmethod
    def _scenario_numeric_id(
        scenario_id,
    ) -> Optional[int]:
        """
        Convert a scenario ID into its numeric component.

        Returns None for invalid IDs.
        """

        if isinstance(
            scenario_id,
            bool,
        ):
            return None

        if isinstance(
            scenario_id,
            int,
        ):

            if scenario_id < 0:
                return None

            return scenario_id

        value = str(
            scenario_id
        ).strip()

        if not value:
            return None

        if value.isdigit():
            return int(value)

        match = re.match(
            r"^scenario_(\d+)$",
            value,
            re.IGNORECASE,
        )

        if match:
            return int(
                match.group(1)
            )

        return None

    # ========================================================================
    # LOGGING
    # ========================================================================

    def _log_loaded_scenario(
        self,
        scenario: Scenario,
    ) -> None:
        """
        Log scenario loading information.

        Logging is intentionally defensive so that a logger
        implementation cannot break scenario loading.
        """

        if self.logger is None:
            return

        try:

            scenario_id = getattr(
                scenario,
                "scenario_id",
                "unknown",
            )

            scenario_path = getattr(
                scenario,
                "scenario_path",
                "unknown",
            )

            description = getattr(
                scenario,
                "scenario_description",
                "",
            )

            self.logger.info(
                f"Loaded scenario: {scenario_id}"
            )

            self.logger.info(
                f"Path: {scenario_path}"
            )

            self.logger.info(
                "Safe scenario text: "
                f"{len(description)} characters"
            )

        except Exception:
            # Logging must never make a valid scenario fail.
            pass


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_scenario(
    dataset_path: Path | str,
    scenario_id: str | int,
) -> Scenario:

    loader = ScenarioLoader(
        dataset_path=dataset_path
    )

    return loader.load_scenario(
        scenario_id=scenario_id
    )


def discover_scenarios(
    dataset_path: Path | str,
) -> List[Path]:

    loader = ScenarioLoader(
        dataset_path=dataset_path
    )

    return loader.discover_scenarios()