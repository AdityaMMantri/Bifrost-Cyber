from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================
# METADATA
# ============================================================

@dataclass
class Metadata:
    """
    Structured representation of metadata.json.

    metadata.json is builder-side ground-truth/control data.
    It is NOT directly exposed to the model.
    """

    scenario_name: str

    relevant_files: list[str] = field(
        default_factory=list
    )

    noise_files: list[str] = field(
        default_factory=list
    )

    technology_stack: list[str] = field(
        default_factory=list
    )

    trust_boundary: Optional[str] = None

    vulnerabilities: list[dict] = field(
        default_factory=list
    )

    optimal_attack: Optional[str] = None

    valid_attacks: list[str] = field(
        default_factory=list
    )

    cwe_reference: Optional[str] = None

    injection_point: Optional[str] = None

    sink: Optional[str] = None

    severity: Optional[str] = None

    description: Optional[str] = None

    requires_cross_file_reasoning: bool = False

    # Preserve complete original metadata for:
    # - validation
    # - debugging
    # - label generation
    # - dataset analysis
    #
    # NEVER directly insert raw_data into the model prompt.
    raw_data: dict = field(
        default_factory=dict
    )


# ============================================================
# SCENARIO
# ============================================================

@dataclass
class Scenario:
    """
    Parsed representation of scenario.md.

    scenario.md is an important part of the SFT input pipeline.

    The loader keeps THREE representations:

    1. sections
       --------------------------------------------------------
       Every section parsed from scenario.md.

       This is builder-side information and may contain
       ground-truth sections such as:

           Expected Attack
           Primary Vulnerability
           Expected Defense
           Attack Flow
           Success Condition

       These MUST NOT automatically be exposed to the model.


    2. neutral_sections
       --------------------------------------------------------
       Only sections explicitly approved as safe contextual
       information.

       These are the sections PromptBuilder uses in the
       model-visible prompt.

       Typical examples:

           Description
           Overview
           Architecture
           Technology Stack
           Normal Workflow
           Business Workflow
           Trust Boundary
           Security Model


    3. raw_markdown
       --------------------------------------------------------
       The original scenario.md contents.

       Kept only for debugging/auditing.

       It MUST NEVER be inserted directly into the model
       prompt because it may contain ground truth.
    """

    scenario_name: str

    # Every parsed section from scenario.md.
    sections: dict[str, str] = field(
        default_factory=dict
    )

    # Only model-safe contextual sections.
    neutral_sections: dict[str, str] = field(
        default_factory=dict
    )

    # Original scenario.md content.
    raw_markdown: str = ""


# ============================================================
# RESOLVED FILE
# ============================================================

@dataclass
class ResolvedFile:
    """
    Represents a source file referenced by metadata.json
    after resolving it to an actual filesystem path.
    """

    relative_path: str

    absolute_path: Path

    # Expected values:
    #
    #   relevant
    #   noise
    #
    file_type: str = "unknown"


# ============================================================
# CODE FILE
# ============================================================

@dataclass
class CodeFile:
    """
    Loaded source-code file.

    These objects are used by PromptBuilder to construct the
    model-visible source-code section.
    """

    relative_path: str

    absolute_path: Path

    # relevant / noise
    file_type: str

    content: str

    line_count: int = 0

    character_count: int = 0


# ============================================================
# RED SFT EXAMPLE
# ============================================================

@dataclass
class RedExample:
    """
    One Red SFT example from red_sft.json.

    IMPORTANT
    ---------
    Red SFT contains ground-truth-oriented information.

    The fields below are builder-side only.

    They may be used to:
        - determine labels
        - validate consistency
        - inspect demonstrations
        - debug dataset construction

    They MUST NOT be inserted into the model-visible
    classification context unless explicitly intended as
    training-answer content.
    """

    example_id: str

    # Original Red SFT user question.
    question: str = ""

    # Original Red SFT assistant answer.
    answer: str = ""

    # Parsed attack label, if present.
    attack: Optional[str] = None

    # Parsed reasoning, if present.
    reasoning: Optional[str] = None

    # Alternative/wrong attack hypotheses.
    wrong_actions: list[str] = field(
        default_factory=list
    )

    # Explanation of why wrong actions fail.
    wrong_action_reasoning: Optional[str] = None


# ============================================================
# VULNERABILITY GROUND TRUTH
# ============================================================

@dataclass
class VulnerabilityGroundTruth:
    """
    Normalized vulnerability information.

    This is BUILDER-SIDE ONLY.

    It must never be inserted into the model-visible
    classification prompt.
    """

    # PrimeVul-style binary classification:
    #
    #   1 = vulnerable
    #   0 = non-vulnerable
    target: int

    vulnerability_type: Optional[str] = None

    cwe: Optional[str] = None

    cve: Optional[str] = None

    description: Optional[str] = None

    attack: Optional[str] = None

    reasoning: Optional[str] = None

    source: Optional[str] = None

    sink: Optional[str] = None


# ============================================================
# PROMPT
# ============================================================

@dataclass
class Prompt:
    """
    Final prompt sent to the model.

    system:
        Contents of system_prompt.txt.

    user:
        Contents of the selected prompt template after
        inserting:

            - neutral scenario.md context
            - source-code files
            - classification question

    Ground-truth metadata must NOT appear here.
    """

    system: str

    user: str


# ============================================================
# PRIMEVUL RECORD
# ============================================================

@dataclass
class PrimeVulRecord:
    """
    One output record in primevul_custom.jsonl.

    The structure contains the important PrimeVul-style
    classification fields.
    """

    project: str

    commit_id: Optional[str]

    target: int

    func: str

    cwe: Optional[str] = None

    cve: Optional[str] = None

    cve_desc: Optional[str] = None

    # Internal scenario identifier.
    #
    # JSONLWriter may omit this from the final PrimeVul-style
    # output depending on the selected output schema.
    scenario_id: Optional[str] = None


# ============================================================
# BUILD STATISTICS
# ============================================================

@dataclass
class BuildStats:
    """
    Statistics accumulated during dataset generation.
    """

    total_scenarios: int = 0

    successful: int = 0

    failed: int = 0

    vulnerable_samples: int = 0

    safe_samples: int = 0

    total_code_files: int = 0

    total_code_characters: int = 0

    total_prompt_characters: int = 0

    errors: list[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Register target
    # --------------------------------------------------------

    def register_target(
        self,
        target: int,
    ) -> None:
        """
        Register one generated sample according to its target.
        """

        if target == 1:

            self.vulnerable_samples += 1

        elif target == 0:

            self.safe_samples += 1

        else:

            raise ValueError(
                f"Invalid target: {target}. "
                f"Expected 0 or 1."
            )

    # --------------------------------------------------------
    # Register successful scenario
    # --------------------------------------------------------

    def register_success(
        self,
    ) -> None:
        """
        Register a successfully processed scenario.
        """

        self.successful += 1

    # --------------------------------------------------------
    # Register error
    # --------------------------------------------------------

    def register_error(
        self,
        error: str,
    ) -> None:
        """
        Register a failed scenario and preserve the error.
        """

        self.failed += 1

        self.errors.append(
            str(error)
        )

    # --------------------------------------------------------
    # Register code
    # --------------------------------------------------------

    def register_code(
        self,
        file_count: int,
        character_count: int,
    ) -> None:
        """
        Register source-code statistics.
        """

        self.total_code_files += (
            file_count
        )

        self.total_code_characters += (
            character_count
        )

    # --------------------------------------------------------
    # Register prompt
    # --------------------------------------------------------

    def register_prompt(
        self,
        character_count: int,
    ) -> None:
        """
        Register generated prompt size.
        """

        self.total_prompt_characters += (
            character_count
        )


# ============================================================
# SCENARIO BUILD RESULT
# ============================================================

@dataclass
class ScenarioBuildResult:
    """
    Result produced after processing one scenario.

    Keeps useful intermediate information together while
    processing a scenario.
    """

    scenario_name: str

    record: Optional[
        PrimeVulRecord
    ] = None

    prompt: Optional[
        Prompt
    ] = None

    ground_truth: Optional[
        VulnerabilityGroundTruth
    ] = None

    success: bool = False

    error: Optional[str] = None