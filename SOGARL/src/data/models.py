"""
models.py

Data structures used throughout the SOGARL pipeline.

This file ONLY defines data structures.

It does NOT:

    - load files
    - build prompts
    - generate model responses
    - calculate Oracle rewards
    - perform GRPO
    - select files
    - perform model inference

Main flow:

    ScenarioLoader
          ↓
    MetadataLoader
          ↓
    PathResolver
          ↓
    CodeLoader
          ↓
    PromptBuilder
          ↓
    Generator
          ↓
    Oracle
          ↓
    RewardManager
          ↓
    GRPO


IMPORTANT INFORMATION BOUNDARY
------------------------------

metadata.json contains ground truth.

Examples:

    - valid attacks
    - valid defenses
    - vulnerabilities
    - root cause
    - optimal attack
    - optimal defense
    - relevant files
    - noise files

This information is required by the Oracle and data-selection
components.

It must NOT be automatically exposed to Red or Blue.

ScenarioContext therefore intentionally does NOT provide a
metadata property.

PromptBuilder must explicitly construct model-visible context
from:

    - safe scenario.md sections
    - selected code files
    - Turn-specific interaction context

and must never serialize the complete Scenario object.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# SCENARIO
# ============================================================================

@dataclass
class Scenario:
    """
    Complete scenario-level information.

    Contains:

        - scenario identification
        - sanitized scenario.md content
        - metadata.json information
        - optionally attached code files

    IMPORTANT:

    The metadata field is ground truth and is intended for:

        - Oracle
        - PathResolver
        - weakness sampling
        - evaluation

    It must NOT be passed directly to Red or Blue.

    Code files are attached after:

        PathResolver
              ↓
        CodeLoader
    """

    scenario_id: str

    scenario_path: Path

    # Only SAFE sections extracted from scenario.md.
    #
    # This is the model-visible scenario description.
    scenario_description: str

    # Complete metadata.json.
    #
    # Oracle/data-side information.
    metadata: Dict[str, Any]

    # Optional runtime code files.
    #
    # Usually populated after PathResolver + CodeLoader.
    code_files: List["CodeFile"] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Metadata shortcuts
    # ------------------------------------------------------------------

    @property
    def name(self) -> Optional[str]:
        return self.metadata.get("name")

    @property
    def category(self) -> Optional[str]:
        return self.metadata.get("category")

    @property
    def attack_type(self) -> Optional[str]:
        return self.metadata.get("attack_type")

    @property
    def difficulty(self) -> Optional[str]:
        return self.metadata.get("difficulty")

    @property
    def scenario_type(self) -> Optional[str]:
        return self.metadata.get("scenario_type")

    @property
    def vulnerabilities(self) -> List[Dict[str, Any]]:
        return self.metadata.get(
            "vulnerabilities",
            []
        )

    @property
    def valid_attacks(self) -> List[str]:
        return self.metadata.get(
            "valid_attacks",
            []
        )

    @property
    def valid_defenses(self) -> List[str]:
        return self.metadata.get(
            "valid_defenses",
            []
        )

    @property
    def optimal_attack(self) -> Optional[str]:
        return self.metadata.get(
            "optimal_attack"
        )

    @property
    def optimal_defense(self) -> Optional[str]:
        return self.metadata.get(
            "optimal_defense"
        )

    @property
    def relevant_files(self) -> List[str]:
        return self.metadata.get(
            "relevant_files",
            []
        )

    @property
    def noise_files(self) -> List[str]:
        return self.metadata.get(
            "noise_files",
            []
        )

    @property
    def tags(self) -> List[str]:
        return self.metadata.get(
            "tags",
            []
        )

    @property
    def codebase_file_count(self) -> int:
        """
        Number of source files currently attached
        to the scenario.
        """

        return len(
            self.code_files
        )


# ============================================================================
# CODE FILE
# ============================================================================

@dataclass
class CodeFile:
    """
    Represents one source-code file loaded into memory.

    file_type is assigned by PathResolver and is either:

        "relevant"

    or:

        "noise"

    No programming-language or file-extension assumptions
    are made here.

    CodeLoader simply reads the selected file and creates
    this object.
    """

    relative_path: str

    absolute_path: Path

    file_type: str

    content: str

    line_count: int

    character_count: int

    # ------------------------------------------------------------------
    # File classification helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Lightweight representation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Return lightweight metadata about the file.

        Source content is intentionally excluded.
        """

        return {
            "relative_path": self.relative_path,
            "absolute_path": str(
                self.absolute_path
            ),
            "file_type": self.file_type,
            "line_count": self.line_count,
            "character_count": self.character_count,
        }


# ============================================================================
# RESOLVED FILE
# ============================================================================

@dataclass
class ResolvedFile:
    """
    Represents a file selected by PathResolver.

    At this stage:

        source code has NOT been loaded.

    PathResolver:
        creates ResolvedFile

    CodeLoader:
        reads ResolvedFile
        ↓
        creates CodeFile
    """

    relative_path: str

    absolute_path: Path

    file_type: str

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Return lightweight information about the resolved file.
        """

        return {
            "relative_path": self.relative_path,
            "absolute_path": str(
                self.absolute_path
            ),
            "file_type": self.file_type,
        }


# ============================================================================
# SCENARIO CONTEXT
# ============================================================================

@dataclass
class ScenarioContext:
    """
    Model-visible runtime context.

    This is the important information-boundary object.

    Contains:

        - sanitized scenario.md content
        - selected source-code files

    Does NOT expose:

        - complete metadata
        - valid attacks
        - valid defenses
        - vulnerabilities
        - root cause
        - optimal attack
        - optimal defense
        - Oracle reward
        - GRPO state

    The complete metadata remains inside Scenario and is
    accessible to the Oracle/data-side components.

    PromptBuilder receives ScenarioContext and explicitly
    constructs the model prompt from this safe information.
    """

    scenario: Scenario

    code_files: List[CodeFile] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Code-file helpers
    # ------------------------------------------------------------------

    @property
    def relevant_code_files(
        self,
    ) -> List[CodeFile]:

        return [
            code_file
            for code_file in self.code_files
            if code_file.is_relevant
        ]

    @property
    def noise_code_files(
        self,
    ) -> List[CodeFile]:

        return [
            code_file
            for code_file in self.code_files
            if code_file.is_noise
        ]

    @property
    def total_files(self) -> int:
        return len(
            self.code_files
        )

    # ------------------------------------------------------------------
    # Model-visible scenario information
    # ------------------------------------------------------------------

    @property
    def scenario_id(self) -> str:
        return self.scenario.scenario_id

    @property
    def description(self) -> str:
        """
        Sanitized scenario.md content.

        Only sections allowed by:

            SAFE_SCENARIO_SECTIONS

        are present here.
        """

        return self.scenario.scenario_description

    # ------------------------------------------------------------------
    # Safe scenario metadata
    # ------------------------------------------------------------------

    @property
    def category(self) -> Optional[str]:
        """
        Category is safe routing/context information.

        It does not expose vulnerability ground truth.
        """

        return self.scenario.category

    @property
    def difficulty(self) -> Optional[str]:
        """
        Difficulty is safe contextual information.
        """

        return self.scenario.difficulty

    @property
    def tags(self) -> List[str]:
        """
        Scenario tags used for sampling/evaluation.

        These should only be included in a model prompt
        if PromptBuilder explicitly decides they are safe.
        """

        return self.scenario.tags


# ============================================================================
# GENERATION RESULT
# ============================================================================

@dataclass
class GenerationResult:
    """
    Result produced by Generator.

    Used for:

        Turn 1 → Red attack generation
        Turn 2 → Blue defense generation
        Turn 3 → Red defense challenge

    This object stores generation information only.

    Oracle scores and GRPO advantages are stored separately.
    """

    response: str

    prompt: str

    model_name: str

    temperature: float

    generation_index: int = 0

    # ------------------------------------------------------------------
    # Turn information
    # ------------------------------------------------------------------

    turn: int = 0

    # Examples:

        # turn 1:
        # "red_attack"

        # turn 2:
        # "blue_defense"

        # turn 3:
        # "red_challenge"

    generation_type: str = ""

    # ------------------------------------------------------------------
    # Model-side metadata
    # ------------------------------------------------------------------

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# SCORED CANDIDATE
# ============================================================================

@dataclass
class ScoredCandidate:
    """
    Candidate response together with reward information.

    Used by RewardManager and GRPO.

    Reward flow:

        Oracle reward
             ↓
        Oracle advantage

        Interaction reward
             ↓
        Interaction advantage

        Advantage fusion
             ↓
        Final advantage


    IMPORTANT:

    Interaction evidence is attached to the candidate that
    was actually tested.

    Therefore:

        Blue D1
        Blue D2
        Blue D3

    can receive interaction rewards.

    An unrelated Red Turn-1 candidate must NOT receive
    that interaction reward.
    """

    response: str

    oracle_reward: float

    candidate_index: int = 0

    # ------------------------------------------------------------------
    # Generation information
    # ------------------------------------------------------------------

    prompt: str = ""

    generation_type: str = ""

    # ------------------------------------------------------------------
    # Interaction signal
    # ------------------------------------------------------------------

    interaction_reward: Optional[float] = None

    # ------------------------------------------------------------------
    # Normalized advantages
    # ------------------------------------------------------------------

    oracle_advantage: Optional[float] = None

    interaction_advantage: Optional[float] = None

    final_advantage: Optional[float] = None

    # ------------------------------------------------------------------
    # Additional evaluation information
    # ------------------------------------------------------------------

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def has_interaction_reward(self) -> bool:
        """
        Whether this candidate was actually tested during
        Turn 3 interaction.
        """

        return (
            self.interaction_reward
            is not None
        )

    @property
    def has_final_advantage(self) -> bool:
        """
        Whether reward fusion has been performed.
        """

        return (
            self.final_advantage
            is not None
        )


# ============================================================================
# INTERACTION RESULT
# ============================================================================

@dataclass
class InteractionResult:
    """
    Result of one Red challenge against one Blue defense.

    This represents Turn 3 interaction evidence.

    Example:

        Blue defense:
            "Add ownership validation."

        Red response:
            "No attack remains."

        Oracle:
            defense closes vulnerability

        Result:

            Red reward  = +1
            Blue reward = +1


    This object is NOT itself a GRPO rollout.

    In v1, Turn 3 does not create a second Red GRPO update.
    """

    defense_candidate_index: int

    red_response: str

    red_reward: float

    blue_reward: float

    # ------------------------------------------------------------------
    # Oracle classification
    # ------------------------------------------------------------------

    outcome_case: Optional[int] = None

    oracle_finding: Optional[str] = None

    # ------------------------------------------------------------------
    # Ground-truth interpretation
    # ------------------------------------------------------------------

    claimed_attack: Optional[str] = None

    remaining_attack: Optional[str] = None

    attack_remains: Optional[bool] = None

    # ------------------------------------------------------------------
    # Additional information
    # ------------------------------------------------------------------

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )