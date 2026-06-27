"""
config.py

Global configuration for the Blue JSONL Dataset Builder.
Only edit values in this file.
"""

from pathlib import Path

PROJECT_ROOT = Path(r"F:\Capstone")
DATASET_ROOT = PROJECT_ROOT / "SFT" / "SFT_Dataset"

# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

BUILDER_ROOT = PROJECT_ROOT / "DatasetBuilder" /"DatasetBuilderBlue"
OUTPUT_DIR = BUILDER_ROOT / "outputs"
DEBUG_DIR = OUTPUT_DIR / "debug"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = BUILDER_ROOT / "logs"
TRAIN_JSONL = OUTPUT_DIR / "train_blue.jsonl"
#VALIDATION_JSONL = OUTPUT_DIR / "validation_blue.jsonl"

# ==========================================================
# DATASET SELECTION
# ==========================================================

# True -> Process every scenario in SFT_Dataset
BUILD_ALL_SCENARIOS = True

# Used only when BUILD_ALL_SCENARIOS = False, useful for testing
SCENARIOS_TO_BUILD = []

# ==========================================================
# VALIDATION SET
# ==========================================================

#VALIDATION_SCENARIOS = []

# ==========================================================
# PROMPTS
# ==========================================================

SYSTEM_PROMPT_FILE = BUILDER_ROOT / "system_prompt.txt"

PROMPT_TEMPLATE_DIR = BUILDER_ROOT / "prompt_templates"

RANDOM_TEMPLATE = True

DEFAULT_TEMPLATE = "template_01.txt"

# ==========================================================
# CODE FILE LOADING
# ==========================================================

# Always include every relevant file listed in metadata.json
INCLUDE_ALL_RELEVANT = True

# Randomly choose these many noise files
MIN_NOISE_FILES = 2
MAX_NOISE_FILES = 5

# Keep relevant files in metadata order
KEEP_RELEVANT_FILE_ORDER = True

# Shuffle noise files before selecting
SHUFFLE_NOISE_FILES = True

# Metadata is considered the source of truth. Every file referenced inside metadata.json is assumed to be source code.
ASSUME_METADATA_FILES_ARE_CODE = True

# ==========================================================
# RANDOMNESS
# ==========================================================

RANDOM_SEED = 42

# ==========================================================
# DEBUG
# ==========================================================

DEBUG_MODE = True

VERBOSE = True

SAVE_DEBUG_PROMPTS = True

#SAVE_BUILD_REPORT = True

#SAVE_MANIFEST = True

PRINT_METADATA_LOADING = True

PRINT_SCENARIO_LOADING = True

PRINT_PATH_RESOLUTION = True

PRINT_CODE_LOADING = True

PRINT_PROMPT_BUILDING = True

PRINT_JSONL_WRITING = True

# ==========================================================
# VALIDATION
# ==========================================================

STOP_ON_FIRST_ERROR = True

CHECK_MISSING_FILES = True

CHECK_EMPTY_FILES = True

CHECK_DUPLICATE_FILES = True

CHECK_EMPTY_SFT = True

# ==========================================================
# PROMPT HEADERS
# ==========================================================

SECTION_DIVIDER = "=" * 80

HEADER_BUSINESS_CONTEXT = "BUSINESS CONTEXT"

HEADER_WORKFLOW = "BUSINESS WORKFLOW"

HEADER_TECH_STACK = "TECHNOLOGY STACK"

HEADER_TRUST_BOUNDARY = "TRUST BOUNDARY"

HEADER_SOURCE_CODE = "SOURCE CODE"

HEADER_QUESTION = "QUESTION"

# ==========================================================
# BUILDER INFORMATION
# ==========================================================

BUILDER_NAME = "Blue JSONL Dataset Builder"

BUILDER_VERSION = "1.0.0"

SAFE_SCENARIO_SECTIONS = {

    "Description",

    "Overview",

    "Business Domain",

    "Technology Stack",

    "Architecture",

    "Business Workflow",

    "Normal Workflow",

    "Normal Flow",

    "Trust Boundary",

    "Security Model",

    "Canonical Security Boundary"

}