from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ============================================================
# METADATA
# ============================================================

@dataclass
class Metadata:
    """
    Builder-side metadata.

    IMPORTANT:
        Metadata is NEVER passed directly to the model prompt.

    This class intentionally preserves the fields expected by:
        - metadata_loader.py
        - build_primevul_jsonl.py
        - path_resolver.py
        - target determination
    """

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    scenario_id: Optional[str] = None
    name: Optional[str] = None

    # --------------------------------------------------------
    # Scenario information
    # --------------------------------------------------------

    category: Optional[str] = None
    attack_type: Optional[str] = None
    difficulty: Optional[str] = None
    scenario_type: Optional[str] = None

    # --------------------------------------------------------
    # Vulnerability / ground truth
    # --------------------------------------------------------

    vulnerability_count: int = 0
    vulnerabilities: list[Any] = field(
        default_factory=list
    )

    optimal_attack: Optional[str] = None
    optimal_defense: Optional[str] = None

    valid_attacks: list[str] = field(
        default_factory=list
    )

    valid_defenses: list[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Security metadata
    # --------------------------------------------------------

    cwe_reference: Any = None
    cve: Optional[str] = None
    cve_desc: Optional[str] = None

    source: Optional[str] = None
    sink: Optional[str] = None
    injection_point: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    trust_boundary: Optional[str] = None

    # --------------------------------------------------------
    # Reasoning / classification metadata
    # --------------------------------------------------------

    requires_cross_file_reasoning: bool = False
    reasoning_depth: Optional[int] = None

    competing_hypotheses: list[Any] = field(
        default_factory=list
    )

    tags: list[str] = field(
        default_factory=list
    )

    classification: dict[str, Any] = field(
        default_factory=dict
    )

    extra: dict[str, Any] = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # File metadata
    # --------------------------------------------------------

    relevant_files: list[str] = field(
        default_factory=list
    )

    noise_files: list[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Original metadata
    # --------------------------------------------------------

    raw_data: dict[str, Any] = field(
        default_factory=dict
    )

    # ========================================================
    # COMPATIBILITY PROPERTIES
    # ========================================================

    @property
    def scenario_name(self) -> str:
        return (
            self.name
            or self.scenario_id
            or ""
        )

    @scenario_name.setter
    def scenario_name(
        self,
        value: str,
    ) -> None:
        self.name = value

    @property
    def commit_id(self) -> Optional[str]:
        value = self.extra.get(
            "commit_id"
        )

        if isinstance(value, str):
            return value

        return None

    @property
    def is_vulnerable(
        self,
    ) -> Optional[bool]:
        """
        Return explicit vulnerability status when available.

        This property is useful for compatibility, but the
        PrimeVulBuilder still performs its own explicit target
        determination.
        """

        # ----------------------------------------------------
        # Explicit boolean classification
        # ----------------------------------------------------

        value = self.classification.get(
            "is_vulnerable"
        )

        if isinstance(value, bool):
            return value

        # ----------------------------------------------------
        # Explicit ground-truth textual classification
        # ----------------------------------------------------

        ground_truth = self.classification.get(
            "ground_truth"
        )

        if isinstance(
            ground_truth,
            str,
        ):

            normalized = (
                ground_truth
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            if normalized in {
                "vulnerable",
                "vulnerability_present",
            }:
                return True

            if normalized in {
                "non_vulnerable",
                "not_vulnerable",
                "safe",
            }:
                return False

        # ----------------------------------------------------
        # Explicit vulnerability objects
        # ----------------------------------------------------

        if (
            self.vulnerability_count > 0
            or self.vulnerabilities
        ):
            return True

        # ----------------------------------------------------
        # Explicit safe scenario type
        # ----------------------------------------------------

        if self.scenario_type:

            normalized = (
                self.scenario_type
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            if normalized in {
                "no_vulnerability",
                "non_vulnerable",
                "not_vulnerable",
                "safe",
            }:
                return False

        return None

    @property
    def target(
        self,
    ) -> Optional[int]:

        vulnerable = self.is_vulnerable

        if vulnerable is True:
            return 1

        if vulnerable is False:
            return 0

        return None


# ============================================================
# SCENARIO MARKDOWN
# ============================================================

@dataclass
class ScenarioSection:
    """
    One parsed section from scenario.md.

    `included=True` means that ScenarioLoader has determined
    that this section is safe to expose to the model.
    """

    heading: str
    content: str
    level: int = 2

    included: bool = False

    exclusion_reason: Optional[str] = None


@dataclass
class ScenarioContext:
    """
    Neutral application context extracted from scenario.md.

    This is the ONLY scenario.md representation that should
    enter the model-visible prompt.
    """

    scenario_id: str

    sections: list[ScenarioSection] = field(
        default_factory=list
    )

    text: str = ""

    loaded: bool = False

    source_file: Optional[str] = None

    # ========================================================
    # Diagnostics
    # ========================================================

    @property
    def section_count(self) -> int:
        return len(
            self.sections
        )

    @property
    def included_section_count(self) -> int:
        return sum(
            1
            for section in self.sections
            if section.included
        )

    @property
    def excluded_section_count(self) -> int:
        return sum(
            1
            for section in self.sections
            if not section.included
        )


@dataclass
class Scenario:
    """
    Canonical scenario representation consumed by
    PromptBuilder.

    ScenarioLoader returns ScenarioContext.

    The builder converts it using:

        Scenario.from_context(...)
    """

    scenario_id: str

    context: Optional[ScenarioContext] = None

    # --------------------------------------------------------
    # Compatibility fields
    # --------------------------------------------------------

    scenario_name: Optional[str] = None

    raw_markdown: str = ""

    sections: list[ScenarioSection] = field(
        default_factory=list
    )

    neutral_sections: Any = field(
        default_factory=dict
    )

    # ========================================================
    # CONSTRUCTION
    # ========================================================

    @classmethod
    def from_context(
        cls,
        context: ScenarioContext,
    ) -> "Scenario":

        neutral_sections = [
            section
            for section in context.sections
            if section.included
        ]

        return cls(
            scenario_id=context.scenario_id,

            scenario_name=context.scenario_id,

            context=context,

            # IMPORTANT:
            #
            # This is the neutral rendered context, NOT the
            # original raw scenario.md.
            #
            # Therefore PromptBuilder can safely use the
            # context without accidentally receiving excluded
            # ground-truth sections.
            raw_markdown=context.text,

            sections=context.sections,

            neutral_sections=neutral_sections,
        )


# ============================================================
# RESOLVED FILE
# ============================================================

@dataclass
class ResolvedFile:
    """
    File path resolved by PathResolver.

    This class was missing from the previous models.py and
    caused:

        ImportError:
        cannot import name 'ResolvedFile'

    PathResolver and CodeLoader both depend on this exact
    interface.
    """

    relative_path: str
    absolute_path: Path
    file_type: str = "unknown"

    @property
    def path(self) -> str:
        return self.relative_path

    @property
    def is_relevant(self) -> bool:
        return (
            self.file_type.lower()
            == "relevant"
        )

    @property
    def is_noise(self) -> bool:
        return (
            self.file_type.lower()
            == "noise"
        )


# ============================================================
# SOURCE CODE
# ============================================================

@dataclass
class SourceFile:
    """
    Loaded source-code file.

    Constructor matches CodeLoader exactly.
    """

    relative_path: str
    absolute_path: Any
    file_type: str
    content: str
    line_count: int
    character_count: int

    @property
    def path(self) -> str:
        return self.relative_path

    @property
    def is_relevant(self) -> bool:
        return (
            self.file_type.lower()
            == "relevant"
        )

    @property
    def is_noise(self) -> bool:
        return (
            self.file_type.lower()
            == "noise"
        )


# Existing code imports CodeFile.
CodeFile = SourceFile


@dataclass
class SourceBundle:
    files: list[SourceFile] = field(
        default_factory=list
    )

    @property
    def file_count(self) -> int:
        return len(
            self.files
        )

    @property
    def character_count(self) -> int:
        return sum(
            file.character_count
            for file in self.files
        )

    @property
    def text(self) -> str:
        return "\n\n".join(
            (
                f"## FILE: {file.relative_path}\n"
                f"{file.content}"
            )
            for file in self.files
        )


# ============================================================
# MODEL INPUT
# ============================================================

@dataclass
class ModelInput:
    """
    Canonical model-visible input.

    Contains ONLY:

        scenario.md neutral context
        +
        relevant source code

    It deliberately contains no:
        metadata
        target
        CWE
        CVE
        Red SFT
        attack labels
        ground truth
    """

    scenario_id: str

    scenario_context: str

    source_code: str

    source_files: list[str] = field(
        default_factory=list
    )

    def combined_text(self) -> str:

        parts: list[str] = []

        if self.scenario_context.strip():

            parts.append(
                "APPLICATION CONTEXT\n\n"
                + self.scenario_context.strip()
            )

        if self.source_code.strip():

            parts.append(
                "SOURCE CODE\n\n"
                + self.source_code.strip()
            )

        return "\n\n".join(
            parts
        )


# ============================================================
# PROMPT
# ============================================================

@dataclass
class Prompt:
    system: str
    user: str

    @property
    def combined(self) -> str:
        return (
            f"{self.system}\n\n"
            f"{self.user}"
        )


@dataclass
class PromptResult:
    """
    Result returned by PromptBuilder.
    """

    scenario_id: str

    prompt: str

    model_input: Optional[ModelInput] = None

    character_count: int = 0

    estimated_tokens: int = 0

    status: str = "OK"

    def __post_init__(
        self,
    ) -> None:
        self.character_count = len(
            self.prompt
        )


# ============================================================
# RED SFT
# ============================================================

@dataclass
class RedSFTExample:
    """
    Internal Red SFT example.

    IMPORTANT:

        Red SFT is builder-side only.

    The fields below match the constructor currently used by
    red_sft_loader.py:

        example_id
        question
        answer
        attack
        reasoning
        wrong_actions
        wrong_action_reasoning
    """

    example_id: str

    question: str

    answer: str

    attack: Optional[str] = None

    reasoning: Optional[str] = None

    wrong_actions: list[str] = field(
        default_factory=list
    )

    wrong_action_reasoning: Optional[str] = None

    @property
    def is_negative(self) -> bool:
        """
        Determine whether this is a negative/counterfactual
        Red example.

        `none` does NOT mean that the model should be trained
        to output nothing.

        It is simply an internal Red SFT classification state.
        """

        if self.attack is None:
            return True

        normalized = (
            self.attack
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        normalized = " ".join(
            normalized.split()
        )

        return normalized in {
            "none",
            "no attack",
            "no vulnerability",
            "not vulnerable",
            "non vulnerable",
            "safe",
            "benign",
            "no exploitable vulnerability",
            "no exploitable vulnerabilities",
        }


@dataclass
class RedSFTBundle:
    examples: list[RedSFTExample] = field(
        default_factory=list
    )

    @property
    def total(self) -> int:
        return len(
            self.examples
        )

    @property
    def positive(self) -> int:
        return sum(
            not example.is_negative
            for example in self.examples
        )

    @property
    def negative(self) -> int:
        return sum(
            example.is_negative
            for example in self.examples
        )


# ============================================================
# PRIMEVUL RECORD
# ============================================================

@dataclass
class PrimeVulRecord:
    """
    Final internal PrimeVul-style record.

    JSONLWriter deliberately serializes only:

        project
        commit_id
        target
        func
        cwe
        cve
        cve_desc

    Internal fields such as scenario_id and scenario_context
    are NOT written to JSONL.
    """

    project: str

    commit_id: Optional[str]

    target: int

    func: str

    cwe: Optional[Any] = None

    cve: Optional[str] = None

    cve_desc: Optional[str] = None

    # --------------------------------------------------------
    # Internal builder diagnostics
    # --------------------------------------------------------

    scenario_id: Optional[str] = None

    scenario_context: Optional[str] = None

    def validate(self) -> None:

        if self.target not in (
            0,
            1,
        ):
            raise ValueError(
                "target must be 0 or 1, "
                f"got {self.target!r}"
            )

        if (
            not isinstance(
                self.project,
                str,
            )
            or not self.project.strip()
        ):
            raise ValueError(
                "project cannot be empty"
            )

        if (
            not isinstance(
                self.func,
                str,
            )
            or not self.func.strip()
        ):
            raise ValueError(
                "func cannot be empty"
            )

        if (
            self.scenario_id is not None
            and not self.scenario_id.strip()
        ):
            raise ValueError(
                "scenario_id cannot be blank"
            )


# ============================================================
# BUILD STATISTICS
# ============================================================

@dataclass
class BuildStats:
    total_scenarios: int = 0

    successful: int = 0

    failed: int = 0

    vulnerable_samples: int = 0

    safe_samples: int = 0

    total_code_files: int = 0

    total_code_characters: int = 0

    total_context_characters: int = 0

    total_prompt_characters: int = 0

    errors: list[str] = field(
        default_factory=list
    )

    def record_success(
        self,
        *,
        target: int,
        source_files: int = 0,
        source_characters: int = 0,
        context_characters: int = 0,
        prompt_characters: int = 0,
    ) -> None:

        self.successful += 1

        if target == 1:
            self.vulnerable_samples += 1
        else:
            self.safe_samples += 1

        self.total_code_files += (
            source_files
        )

        self.total_code_characters += (
            source_characters
        )

        self.total_context_characters += (
            context_characters
        )

        self.total_prompt_characters += (
            prompt_characters
        )

    def record_failure(
        self,
        message: str,
    ) -> None:

        self.failed += 1

        self.errors.append(
            message
        )

    @property
    def output_records(self) -> int:
        return self.successful


# ============================================================
# SCENARIO BUILD RESULT
# ============================================================

@dataclass
class ScenarioBuildResult:
    scenario_id: str

    metadata: Optional[Metadata] = None

    scenario_context: Optional[
        ScenarioContext
    ] = None

    source_bundle: Optional[
        SourceBundle
    ] = None

    red_sft: Optional[
        RedSFTBundle
    ] = None

    model_input: Optional[
        ModelInput
    ] = None

    prompt_result: Optional[
        PromptResult
    ] = None

    record: Optional[
        PrimeVulRecord
    ] = None

    success: bool = False

    error: Optional[str] = None