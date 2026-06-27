"""
models.py

Shared data models used throughout the Dataset Builder.

Every loader returns one of these models.
Every builder consumes one of these models.

No module should create its own dataclasses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict


# ==========================================================
# Metadata
# ==========================================================

@dataclass
class Metadata:
    """
    Information loaded from metadata.json.
    """

    scenario_name: str

    relevant_files: List[str] = field(default_factory=list)

    noise_files: List[str] = field(default_factory=list)

    competing_hypotheses: List[str] =field(default_factory=list)

    technology_stack: List[str] = field(default_factory=list)

    trust_boundary: str = ""

    raw_data: Dict = field(default_factory=dict)


# ==========================================================
# Scenario
# ==========================================================

@dataclass
class Scenario:
    """
    Parsed representation of scenario.md.

    The loader extracts every markdown section into a generic
    dictionary instead of assuming a fixed schema.

    PromptBuilder later decides which sections are safe to
    expose to the LLM.
    """

    scenario_name: str

    # Original markdown (useful for debugging)
    raw_markdown: str

    # Parsed markdown sections
    # Example:
    #
    # {
    #     "Description": "...",
    #     "Technology Stack": "...",
    #     "Business Workflow": "...",
    #     "Expected Attack": "...",
    #     "Primary Vulnerability": "..."
    # }
    #
    sections: Dict[str, str] = field(default_factory=dict)


# ==========================================================
# Code File
# ==========================================================

@dataclass
class CodeFile:
    """
    Represents one source file included in the prompt.
    """

    relative_path: str

    absolute_path: Path

    file_type: str          # relevant / noise

    content: str

    line_count: int

    character_count: int


# ==========================================================
# SFT Example
# ==========================================================

@dataclass
class SFTExample:
    """
    One example from red_sft.json.
    """

    example_id: str

    question: str

    answer: str


# ==========================================================
# Prompt
# ==========================================================

@dataclass
class Prompt:
    """
    Final prompt sent to the model.
    """

    system: str

    user: str


# ==========================================================
# Dataset Statistics
# ==========================================================

@dataclass
class BuildStats:

    scenarios_processed: int = 0

    training_examples: int = 0

    files_loaded: int = 0

    relevant_files: int = 0

    noise_files: int = 0

    total_prompt_characters: int = 0

    total_prompt_tokens: int = 0

    warnings: int = 0

    errors: int = 0


# ==========================================================
# Resolved File
# ==========================================================

@dataclass
class ResolvedFile:
    """
    Output of PathResolver.
    """

    relative_path: str

    absolute_path: Path

    file_type: str          # relevant / noise