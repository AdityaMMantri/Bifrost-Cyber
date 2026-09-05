from pathlib import Path
import random

PROJECT_ROOT = Path(r"F:\SEM-7\Capstone\Classification\PrimeVulBuilder")
DATASET_ROOT = Path(r"F:\SEM-7\Capstone\SFT\SFT_Dataset")
OUTPUT_DIR = (PROJECT_ROOT/"output")
OUTPUT_FILE = (OUTPUT_DIR/"primevul_custom.jsonl")

# System-level instructions for the vulnerability
# classification model.
SYSTEM_PROMPT_FILE = (PROJECT_ROOT/"system_prompt.txt")
# Per-scenario user prompt template.
PROMPT_TEMPLATE_FILE = (PROJECT_ROOT/"prompt_template.txt")


# Use ONLY files explicitly marked as relevant in metadata.json.
# This means we are NOT performing file-relevance retrieval.
INCLUDE_RELEVANT_FILES_ONLY = False


# ============================================================
# SCENARIO CONTEXT SETTINGS
# ============================================================

# Include neutral application context from scenario.md.
INCLUDE_SCENARIO_CONTEXT = True

MIN_NOISE_FILES = 2
MAX_NOISE_FILES = 5
RANDOM_SEED = 42

# These sections help the classifier understand the software
# without directly giving away the vulnerability.
# These values are passed into ScenarioLoader.
ALLOWED_SCENARIO_SECTIONS = [
    "Description",
    "Overview",
    "Application Overview",
    "Application Description",
    "Technology Stack",
    "Technologies",
    "Architecture",
    "System Architecture",
    "Business Workflow",
    "Workflow",
    "Normal Workflow",
    "Application Workflow",
    "Trust Boundary",
    "Security Model",
    "Security Architecture",
]

# These sections can contain ground-truth information.
# They are NEVER inserted into the classification prompt.
EXCLUDED_SCENARIO_SECTIONS = [
    "Expected Attack",
    "Optimal Attack",
    "Primary Vulnerability",
    "Vulnerability",
    "Known Vulnerability",
    "CWE",
    "CWE Reference",
    "Injection Point",
    "Sink",
    "Source",
    "Attack Path",
    "Attack Chain",
    "Attack Preconditions",
    "Broken Assumption",
    "Optimal Defense",
    "Valid Attacks",
    "Valid Defenses",
    "Ground Truth",
    "Security Finding",
    "Finding",
    "Exploit",
    "Exploitation",
]

# red_sft.json is read internally.
# It is used to:
#   - extract Red-derived attack information;
#   - validate scenario consistency;
#   - help determine ground truth.
# It is NEVER included in the model prompt.
USE_RED_SFT = True
# blue_sft.json is deliberately not used for this
# vulnerability-classification dataset.
# It may exist inside the scenario directory, but the builder does not use it.
USE_BLUE_SFT = False

# PRIMEVUL-STYLE RECORD
# Fields written by jsonl_writer.py.
PRIMEVUL_FIELDS = [
    "project",
    "commit_id",
    "target",
    "func",
    "cwe",
    "cve",
    "cve_desc",
]

# PrimeVul-style binary classification.
# 1 = vulnerable
# 0 = not vulnerable
VULNERABLE_LABEL = 1
NOT_VULNERABLE_LABEL = 0

# custom scenarios do not necessarily have Git commit IDs
# or CVE IDs corresponding to the PrimeVul dataset.
# Therefore these remain None unless legitimate values are
# available.
DEFAULT_COMMIT_ID = None
DEFAULT_CVE = None
# Save the generated classification prompts so you can inspect
# exactly what the model will receive.
SAVE_DEBUG_PROMPTS = True
DEBUG_DIR = (OUTPUT_DIR/"debug")
# Estimate prompt size for multi-file scenarios.
ENABLE_CONTEXT_ESTIMATION = True
# Process scenarios in deterministic alphabetical order.
SORT_SCENARIOS = True
# If one scenario fails:
# True:
#     log the error and continue with remaining scenarios.
# False:
#     stop the entire build immediately.
CONTINUE_ON_ERROR = True

# Create output directory automatically.
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
LOG_DIR = OUTPUT_DIR/"logs"
LOG_DIR.mkdir(parents=True,exist_ok=True)
LOG_FILE = LOG_DIR/"build.log"

if SAVE_DEBUG_PROMPTS:
    DEBUG_DIR.mkdir(parents=True,exist_ok=True)
if SAVE_DEBUG_PROMPTS:
    DEBUG_DIR.mkdir(parents=True,exist_ok=True)