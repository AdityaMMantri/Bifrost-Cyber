from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .models import ScenarioContext, ScenarioSection


class ScenarioLoader:
    """
    Loads scenario.md and produces the model-visible application
    context.

    Architecture
    ------------

        scenario.md
             |
             v
        ScenarioLoader
             |
             +--> remove explicit answer sections
             |
             +--> reject obvious answer leakage
             |
             v
        ScenarioContext
             |
             v
        PromptBuilder
             |
             v
        JSONL `func`

    IMPORTANT
    ---------

    metadata.json and red_sft.json are NOT read here.

    This loader is responsible only for scenario.md.

    Ground-truth fields such as:

        - target
        - optimal_attack
        - optimal_defense
        - CWE
        - ground_truth
        - expected classification
        - attack path
        - remediation

    must never be included in the model-visible context.
    """

    # ============================================================
    # EXPLICITLY EXCLUDED HEADINGS
    # ============================================================

    EXCLUDED_HEADINGS = {
        "classification",
        "classification label",
        "scenario type",
        "expected classifier output",
        "expected classification",
        "ground truth",
        "ground truth label",
        "ground truth classification",
        "vulnerability",
        "vulnerabilities",
        "vulnerability details",
        "attack",
        "attacks",
        "attack path",
        "attack paths",
        "expected attack",
        "expected attacks",
        "optimal attack",
        "optimal attacks",
        "optimal defense",
        "optimal defenses",
        "valid attacks",
        "valid defenses",
        "competing hypotheses",
        "exploit",
        "exploits",
        "exploit path",
        "exploit paths",
        "exploitation",
        "red team",
        "red sft",
        "blue team",
        "blue sft",
        "labels",
        "dataset labels",
        "security verdict",
        "security classification",
        "expected remediation",
        "remediation",
        "defense",
        "defenses",
        "countermeasures",
    }

    # ============================================================
    # SECURITY-CONCLUSION HEADINGS
    # ============================================================

    SECURITY_CONCLUSION_HEADINGS = {
        "security properties",
        "security invariants",
        "security guarantees",
        "security requirements",
        "security controls",
        "prohibited vulnerabilities",
        "attack surface",
        "expected attack path",
        "expected remediation",
        "root cause",
        "impact",
        "remediation",
        "defense",
        "defenses",
        "countermeasures",
        "expected security outcome",
        "security outcome",
        "ground truth",
    }

    # ============================================================
    # NORMAL APPLICATION-CONTEXT HEADINGS
    # ============================================================

    ALLOWED_HEADINGS = {
        "overview",
        "description",
        "components",
        "component",
        "technology stack",
        "technology",
        "technologies",
        "languages",
        "language",
        "frameworks",
        "framework",
        "architecture",
        "workflow",
        "core workflow",
        "process",
        "trust boundary",
        "trust boundaries",
        "security model",
        "system model",
        "application model",
        "data flow",
        "request flow",
        "deployment model",
        "environment",
        "business context",
        "business domain",
        "domain",
        "interfaces",
        "services",
        "service",
        "data model",
        "dependencies",
        "configuration",
        "modules",
        "module",
        "operations",
        "operation",
        "api",
        "apis",
        "inputs",
        "outputs",
        "storage",
        "database",
        "databases",
        "integration",
        "integrations",
        "system behavior",
        "application behavior",
        "workflow steps",
    }

    # ============================================================
    # CONTENT-LEVEL LEAKAGE MARKERS
    #
    # These are deliberately explicit. We do NOT reject ordinary
    # words such as "attack" when they occur naturally in prose.
    # ============================================================

    ANSWER_LEAKAGE_PATTERNS = [
        r"\bexpected\s+classifier\s+output\s*:",
        r"\bexpected\s+classification\s*:",
        r"\bclassification\s+label\s*:",
        r"\bground[_\s-]*truth\s*:",
        r"\boptimal[_\s-]*attack\s*:",
        r"\boptimal[_\s-]*defense\s*:",
        r"\bvalid[_\s-]*attacks\s*:",
        r"\bvalid[_\s-]*defenses\s*:",
        r"\bexpected[_\s-]*attack\s*:",
        r"\bexpected[_\s-]*defense\s*:",
        r"\bexpected[_\s-]*remediation\s*:",
        r"\bdataset[_\s-]*labels?\s*:",
        r"\bsecurity\s+verdict\s*:",
        r"\bsecurity\s+classification\s*:",
        r"\broot[_\s-]*cause\s*:",
        r"\battack[_\s-]*path\s*:",
        r"\bexploit[_\s-]*path\s*:",
        r"\bno[_\s-]*exploitable[_\s-]*vulnerabilit(?:y|ies)\b",
        r"\bno[_\s-]*vulnerability\b",
        r"\bnon[_\s-]*vulnerable\b",
        r"\bis[_\s-]*vulnerable\s*:",
        r"\bvulnerability[_\s-]*count\s*:",
    ]

    # ============================================================
    # CONSTRUCTOR
    # ============================================================

    def __init__(
        self,
        *,
        include_allowed_only: bool = True,
        reject_content_leakage: bool = True,
    ) -> None:

        self.include_allowed_only = (
            include_allowed_only
        )

        self.reject_content_leakage = (
            reject_content_leakage
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def load(
        self,
        scenario_dir: Path,
        scenario_id: Optional[str] = None,
    ) -> ScenarioContext:
        """
        Load scenario.md from the supplied scenario directory.

        The resulting ScenarioContext contains ONLY the neutral
        scenario context selected from scenario.md.
        """

        scenario_dir = Path(
            scenario_dir
        )

        if scenario_id is None:
            scenario_id = scenario_dir.name

        scenario_file = (
            scenario_dir / "scenario.md"
        )

        if not scenario_file.exists():

            raise FileNotFoundError(
                f"scenario.md not found: "
                f"{scenario_file}"
            )

        if not scenario_file.is_file():

            raise ValueError(
                f"scenario.md is not a regular file: "
                f"{scenario_file}"
            )

        raw_text = scenario_file.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if not raw_text.strip():

            raise ValueError(
                f"scenario.md is empty: "
                f"{scenario_file}"
            )

        sections = self._parse_sections(
            raw_text
        )

        if not sections:

            raise ValueError(
                f"scenario.md contains no Markdown "
                f"sections: {scenario_file}"
            )

        selected_sections = (
            self._select_neutral_sections(
                sections
            )
        )

        if not selected_sections:

            raise ValueError(
                f"No neutral application-context "
                f"sections found in scenario.md: "
                f"{scenario_file}"
            )

        # --------------------------------------------------------
        # Content-level leakage check.
        #
        # If the surviving context contains an explicit answer
        # marker, FAIL rather than silently producing a poisoned
        # training sample.
        # --------------------------------------------------------

        if self.reject_content_leakage:

            self._validate_no_answer_leakage(
                selected_sections,
                scenario_file,
            )

        context_text = (
            self._render_context(
                selected_sections
            )
        )

        if not context_text.strip():

            raise ValueError(
                f"scenario.md produced empty "
                f"model-visible context: "
                f"{scenario_file}"
            )

        return ScenarioContext(
            scenario_id=scenario_id,
            sections=selected_sections,
            text=context_text,
            loaded=True,
            source_file=str(
                scenario_file
            ),
        )

    # ============================================================
    # MARKDOWN PARSING
    # ============================================================

    def _parse_sections(
        self,
        text: str,
    ) -> list[ScenarioSection]:
        """
        Parse Markdown ATX headings.

        Supports:

            # Heading
            ## Heading
            ### Heading
            ...

        Content continues until the next heading.
        """

        lines = text.splitlines()

        sections: list[ScenarioSection] = []

        current_heading: Optional[str] = None

        current_level = 2

        current_content: list[str] = []

        def flush() -> None:

            nonlocal current_heading
            nonlocal current_level
            nonlocal current_content

            if current_heading is None:
                return

            content = (
                "\n".join(
                    current_content
                ).strip()
            )

            sections.append(
                ScenarioSection(
                    heading=current_heading,
                    content=content,
                    level=current_level,
                )
            )

            current_heading = None
            current_content = []

        for line in lines:

            heading_match = re.match(
                r"^\s*(#{1,6})\s+(.+?)\s*$",
                line,
            )

            if heading_match:

                flush()

                hashes = (
                    heading_match.group(1)
                )

                heading = (
                    heading_match.group(2)
                )

                heading = re.sub(
                    r"\s+#+\s*$",
                    "",
                    heading,
                ).strip()

                current_heading = (
                    heading
                )

                current_level = len(
                    hashes
                )

            else:

                current_content.append(
                    line
                )

        flush()

        return sections

    # ============================================================
    # HEADING NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_heading(
        heading: str,
    ) -> str:
        """
        Normalize Markdown heading text for comparison.
        """

        normalized = (
            heading
            .strip()
            .lower()
        )

        # Remove Markdown formatting.
        normalized = normalized.replace(
            "`",
            "",
        )

        normalized = normalized.replace(
            "*",
            "",
        )

        # Underscore is commonly used in metadata-style
        # headings and should behave like whitespace.
        normalized = normalized.replace(
            "_",
            " ",
        )

        normalized = normalized.replace(
            "-",
            " ",
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    # ============================================================
    # SECTION SELECTION
    # ============================================================

    def _select_neutral_sections(
        self,
        sections: list[ScenarioSection],
    ) -> list[ScenarioSection]:
        """
        Select model-safe application-context sections.

        Explicit answer/security-conclusion sections are always
        excluded.

        Unknown headings are excluded when
        include_allowed_only=True.
        """

        selected: list[
            ScenarioSection
        ] = []

        for section in sections:

            heading = (
                self._normalize_heading(
                    section.heading
                )
            )

            # ----------------------------------------------------
            # Empty section
            # ----------------------------------------------------

            if not section.content.strip():

                section.included = False

                section.exclusion_reason = (
                    "empty section"
                )

                continue

            # ----------------------------------------------------
            # Explicit ground-truth section
            # ----------------------------------------------------

            if (
                heading
                in self.EXCLUDED_HEADINGS
            ):

                section.included = False

                section.exclusion_reason = (
                    "ground-truth heading"
                )

                continue

            # ----------------------------------------------------
            # Security conclusion
            # ----------------------------------------------------

            if (
                heading
                in self.SECURITY_CONCLUSION_HEADINGS
            ):

                section.included = False

                section.exclusion_reason = (
                    "security-conclusion heading"
                )

                continue

            # ----------------------------------------------------
            # Allowed neutral section
            # ----------------------------------------------------

            if (
                self.include_allowed_only
                and heading
                not in self.ALLOWED_HEADINGS
            ):

                section.included = False

                section.exclusion_reason = (
                    "heading not in neutral allowlist"
                )

                continue

            # ----------------------------------------------------
            # Select
            # ----------------------------------------------------

            section.included = True

            section.exclusion_reason = None

            selected.append(
                section
            )

        return selected

    # ============================================================
    # CONTENT LEAKAGE VALIDATION
    # ============================================================

    def _validate_no_answer_leakage(
        self,
        sections: list[ScenarioSection],
        scenario_file: Path,
    ) -> None:
        """
        Reject explicit answer-bearing content that survived
        section filtering.

        This is intentionally conservative for dataset creation:
        failing a scenario is preferable to silently creating a
        label-leaked training sample.
        """

        findings: list[str] = []

        for section in sections:

            for pattern in (
                self.ANSWER_LEAKAGE_PATTERNS
            ):

                match = re.search(
                    pattern,
                    section.content,
                    re.IGNORECASE,
                )

                if match:

                    findings.append(
                        f"{section.heading!r}: "
                        f"{match.group(0)!r}"
                    )

        if findings:

            raise ValueError(
                "Answer leakage detected in "
                f"scenario.md: {scenario_file}. "
                "Remove ground-truth/classification "
                "information from the neutral scenario "
                "context. Findings: "
                + "; ".join(findings)
            )

    # ============================================================
    # CONTEXT RENDERING
    # ============================================================

    def _render_context(
        self,
        sections: list[ScenarioSection],
    ) -> str:
        """
        Render only selected neutral sections.
        """

        blocks: list[str] = []

        for section in sections:

            content = (
                section.content.strip()
            )

            if not content:
                continue

            heading_prefix = "#" * max(
                2,
                min(
                    section.level,
                    6,
                ),
            )

            blocks.append(
                f"{heading_prefix} "
                f"{section.heading.strip()}\n\n"
                f"{content}"
            )

        return (
            "\n\n".join(
                blocks
            ).strip()
        )

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    def get_diagnostics(
        self,
        context: ScenarioContext,
    ) -> dict:
        """
        Return builder/debug information.

        This is for debug logs only.

        Do NOT place the diagnostics object into ModelInput.
        """

        return {
            "scenario_id": (
                context.scenario_id
            ),
            "scenario_md_loaded": (
                context.loaded
            ),
            "scenario_md": (
                context.source_file
            ),
            "total_sections": (
                context.section_count
            ),
            "included_sections": (
                context.included_section_count
            ),
            "excluded_sections": (
                context.excluded_section_count
            ),
            "context_characters": (
                len(context.text)
            ),
            "sections": [
                {
                    "heading": (
                        section.heading
                    ),
                    "level": (
                        section.level
                    ),
                    "included": (
                        section.included
                    ),
                    "reason": (
                        section.exclusion_reason
                    ),
                    "characters": (
                        len(section.content)
                    ),
                }
                for section in context.sections
            ],
        }