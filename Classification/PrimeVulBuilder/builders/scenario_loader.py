from pathlib import Path

from .models import Scenario


class ScenarioLoader:
    """
    Loads and parses scenario.md for the PrimeVul builder.

    Responsibilities
    ----------------
    - Read scenario.md
    - Parse Markdown sections
    - Identify neutral application context
    - Exclude vulnerability/ground-truth information
    - Return a Scenario object

    Important
    ---------
    The complete scenario.md is NOT automatically exposed to
    the model.

    Only sections explicitly allowed by the builder configuration
    are included in Scenario.neutral_sections.

    Ground-truth sections such as:
        - Expected Attack
        - Vulnerability
        - CWE
        - Injection Point
        - Sink
        - Attack Path
        - Ground Truth

    are never included in model-visible context.
    """

    # ============================================================
    # DEFAULT SAFE SECTIONS
    # ============================================================

    DEFAULT_ALLOWED_SECTIONS = {
        "description",
        "overview",
        "application overview",
        "application description",
        "technology stack",
        "technologies",
        "architecture",
        "system architecture",
        "business workflow",
        "workflow",
        "normal workflow",
        "application workflow",
        "trust boundary",
        "security model",
        "security architecture",
    }

    # ============================================================
    # DEFAULT EXCLUDED SECTIONS
    # ============================================================

    DEFAULT_EXCLUDED_SECTIONS = {
        "expected attack",
        "optimal attack",
        "primary vulnerability",
        "vulnerability",
        "known vulnerability",
        "cwe",
        "cwe reference",
        "injection point",
        "sink",
        "source",
        "attack path",
        "attack chain",
        "attack preconditions",
        "broken assumption",
        "optimal defense",
        "valid attacks",
        "valid defenses",
        "ground truth",
        "security finding",
        "finding",
        "exploit",
        "exploitation",
    }

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        logger=None,
        allowed_sections=None,
        excluded_sections=None,
    ):
        """
        Parameters
        ----------
        logger:
            Optional Logger instance.

        allowed_sections:
            Sections that are allowed to become model-visible
            application context.

            If None, DEFAULT_ALLOWED_SECTIONS is used.

        excluded_sections:
            Sections that must never become model-visible.

            If None, DEFAULT_EXCLUDED_SECTIONS is used.

        Explicit exclusions ALWAYS take priority over allowed
        sections.
        """

        self.logger = logger

        if allowed_sections is None:
            allowed_sections = (
                self.DEFAULT_ALLOWED_SECTIONS
            )

        if excluded_sections is None:
            excluded_sections = (
                self.DEFAULT_EXCLUDED_SECTIONS
            )

        self.allowed_sections = {
            self._normalize_heading(
                heading
            )
            for heading in allowed_sections
        }

        self.excluded_sections_set = {
            self._normalize_heading(
                heading
            )
            for heading in excluded_sections
        }

    # ============================================================
    # PUBLIC API
    # ============================================================

    def load(
        self,
        scenario_path: Path,
    ) -> Scenario:
        """
        Load scenario.md from a scenario directory.

        Parameters
        ----------
        scenario_path:
            Path to the scenario directory.

        Returns
        -------
        Scenario
            Parsed scenario containing:
                - all sections
                - neutral model-safe sections
                - original Markdown
        """

        scenario_path = Path(
            scenario_path
        )

        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Scenario directory not found: "
                f"{scenario_path}"
            )

        if not scenario_path.is_dir():
            raise ValueError(
                f"Scenario path is not a directory: "
                f"{scenario_path}"
            )

        scenario_file = (
            scenario_path / "scenario.md"
        )

        if not scenario_file.exists():
            raise FileNotFoundError(
                f"scenario.md not found in: "
                f"{scenario_path}"
            )

        if not scenario_file.is_file():
            raise ValueError(
                f"scenario.md is not a file: "
                f"{scenario_file}"
            )

        raw_markdown = self._read_file(
            scenario_file
        )

        sections = self._parse_sections(
            raw_markdown
        )

        neutral_sections = (
            self._select_neutral_sections(
                sections
            )
        )

        scenario = Scenario(
            scenario_name=scenario_path.name,
            sections=sections,
            neutral_sections=neutral_sections,
            raw_markdown=raw_markdown,
        )

        if self.logger:

            self.logger.info(
                f"Loaded scenario: "
                f"{scenario_path.name}"
            )

            self.logger.info(
                f"Total scenario sections: "
                f"{len(sections)}"
            )

            self.logger.info(
                f"Neutral sections included: "
                f"{len(neutral_sections)}"
            )

            excluded = (
                self.excluded_sections(
                    scenario
                )
            )

            if excluded:
                self.logger.info(
                    "Excluded scenario sections: "
                    + ", ".join(excluded)
                )

        return scenario

    # ============================================================
    # FILE READING
    # ============================================================

    def _read_file(
        self,
        path: Path,
    ) -> str:
        """
        Read scenario.md as UTF-8 text.
        """

        try:

            return path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError as exc:

            raise ValueError(
                f"Could not decode scenario file "
                f"as UTF-8: {path}"
            ) from exc

        except OSError as exc:

            raise OSError(
                f"Could not read scenario file: "
                f"{path}\n{exc}"
            ) from exc

    # ============================================================
    # MARKDOWN PARSER
    # ============================================================

    def _parse_sections(
        self,
        markdown: str,
    ) -> dict[str, str]:
        """
        Parse Markdown headings into a dictionary.

        Supports:

            # Heading
            ## Heading
            ### Heading

        Content continues until the next Markdown heading.

        Example:

            ## Architecture

            The backend contains...

            ## Workflow

            The user first...

        becomes:

            {
                "Architecture":
                    "The backend contains...",

                "Workflow":
                    "The user first..."
            }
        """

        sections: dict[str, str] = {}

        current_heading = None
        current_content: list[str] = []

        lines = markdown.splitlines()

        for line in lines:

            stripped = line.strip()

            # ----------------------------------------------------
            # Markdown heading
            # ----------------------------------------------------

            if stripped.startswith("#"):

                heading = (
                    stripped
                    .lstrip("#")
                    .strip()
                )

                if not heading:
                    continue

                # Save previous section.
                if current_heading is not None:

                    content = (
                        self._clean_content(
                            current_content
                        )
                    )

                    sections[
                        current_heading
                    ] = content

                current_heading = heading
                current_content = []

            else:

                if current_heading is not None:
                    current_content.append(
                        line
                    )

        # --------------------------------------------------------
        # Save final section.
        # --------------------------------------------------------

        if current_heading is not None:

            content = (
                self._clean_content(
                    current_content
                )
            )

            sections[
                current_heading
            ] = content

        return sections

    # ============================================================
    # CONTENT CLEANING
    # ============================================================

    def _clean_content(
        self,
        lines: list[str],
    ) -> str:
        """
        Remove excessive blank lines while preserving
        scenario content.
        """

        cleaned = []

        previous_blank = False

        for line in lines:

            line = line.rstrip()

            if not line.strip():

                if not previous_blank:
                    cleaned.append("")

                previous_blank = True

            else:

                cleaned.append(
                    line
                )

                previous_blank = False

        return "\n".join(
            cleaned
        ).strip()

    # ============================================================
    # SECTION NORMALIZATION
    # ============================================================

    def _normalize_heading(
        self,
        heading: str,
    ) -> str:
        """
        Normalize a heading before comparison.

        Examples:

            "Architecture:"
            "architecture"
            "  ARCHITECTURE  "

        all become:

            "architecture"
        """

        normalized = (
            heading
            .lower()
            .strip()
        )

        normalized = normalized.rstrip(
            ":.-_"
        )

        normalized = " ".join(
            normalized.split()
        )

        return normalized

    # ============================================================
    # SECTION FILTERING
    # ============================================================

    def _select_neutral_sections(
        self,
        sections: dict[str, str],
    ) -> dict[str, str]:
        """
        Select only model-safe contextual sections.

        Filtering order:

            1. Normalize heading.
            2. Check explicit exclusion.
            3. Check allowed section.
            4. Ignore empty sections.

        Explicit exclusion ALWAYS wins.
        """

        neutral_sections: dict[
            str, str
        ] = {}

        for heading, content in (
            sections.items()
        ):

            normalized = (
                self._normalize_heading(
                    heading
                )
            )

            # ----------------------------------------------------
            # Security rule:
            #
            # Explicit exclusion ALWAYS wins.
            # ----------------------------------------------------

            if (
                normalized
                in self.excluded_sections_set
            ):
                continue

            # ----------------------------------------------------
            # Only explicitly allowed contextual sections
            # are model-visible.
            # ----------------------------------------------------

            if (
                normalized
                not in self.allowed_sections
            ):
                continue

            if not content.strip():
                continue

            neutral_sections[
                heading
            ] = content

        return neutral_sections

    # ============================================================
    # PROMPT CONTEXT
    # ============================================================

    def build_context(
        self,
        scenario: Scenario,
    ) -> str:
        """
        Convert neutral sections into the context block used
        by prompt_template.txt.
        """

        if not scenario.neutral_sections:

            return (
                "No additional application "
                "context was provided."
            )

        blocks = []

        for heading, content in (
            scenario.neutral_sections.items()
        ):

            blocks.append(
                f"## {heading}\n\n"
                f"{content}"
            )

        return "\n\n".join(
            blocks
        )

    # ============================================================
    # INSPECTION HELPERS
    # ============================================================

    def get_all_sections(
        self,
        scenario: Scenario,
    ) -> dict[str, str]:
        """
        Return every parsed scenario section.

        For debugging only.

        Do NOT pass this directly to the model.
        """

        return dict(
            scenario.sections
        )

    def get_neutral_sections(
        self,
        scenario: Scenario,
    ) -> dict[str, str]:
        """
        Return only approved model-visible sections.
        """

        return dict(
            scenario.neutral_sections
        )

    def excluded_sections(
        self,
        scenario: Scenario,
    ) -> list[str]:
        """
        Return all parsed sections that were excluded.

        This includes:
            - explicitly excluded sections
            - sections not present in the allowed list
        """

        excluded = []

        for heading in (
            scenario.sections
        ):

            normalized = (
                self._normalize_heading(
                    heading
                )
            )

            if (
                normalized
                in self.excluded_sections_set
            ):
                excluded.append(
                    heading
                )
                continue

            if (
                normalized
                not in self.allowed_sections
            ):
                excluded.append(
                    heading
                )

        return excluded

    # ============================================================
    # SAFETY CHECK
    # ============================================================

    def is_model_safe_heading(
        self,
        heading: str,
    ) -> bool:
        """
        Return True if a heading is safe to expose.

        Explicit exclusions always override allowed sections.
        """

        normalized = (
            self._normalize_heading(
                heading
            )
        )

        if (
            normalized
            in self.excluded_sections_set
        ):
            return False

        return (
            normalized
            in self.allowed_sections
        )