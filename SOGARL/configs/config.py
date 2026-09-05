"""
config.py

Global configuration for SOGARL.

SOGARL
------
Sequential Oracle-Guided Adversarial Reinforcement Learning

This configuration controls:

    1. Dataset paths
    2. Safe scenario context
    3. SFT LoRA adapters
    4. Base / Oracle models
    5. Local merged models
    6. Train / validation / test split
    7. Episode generation
    8. GRPO
    9. Reward fusion
    10. Confidence gating
    11. Oracle
    12. Weakness-based sampling
    13. Testing
    14. Checkpointing
    15. Hardware
    16. Logging

Only configuration values should be changed here.
Training logic belongs inside src/.
"""

from pathlib import Path
import os

SOGARL_ROOT = Path(__file__).resolve().parents[1]
CAPSTONE_ROOT = SOGARL_ROOT.parent

SOGARL_RUNTIME = os.getenv("SOGARL_RUNTIME", "auto").strip().lower()
if SOGARL_RUNTIME not in {"auto", "kaggle", "local"}:
    raise ValueError("SOGARL_RUNTIME must be one of: auto, kaggle, local.")
if SOGARL_RUNTIME == "auto":
    SOGARL_RUNTIME = "kaggle" if Path("/kaggle").exists() else "local"

# Kept with its original name. Override for any SFT location.
SFT_ROOT = Path(os.getenv("SOGARL_SFT_ROOT", str(CAPSTONE_ROOT / "SFT"))).expanduser()


# ============================================================================
# DATASET
# ============================================================================

DATASET_PATH = Path(os.getenv("SOGARL_DATASET_PATH",str(SFT_ROOT / "SFT_Dataset")))

# ============================================================================
# SAFE SCENARIO CONTEXT
# ============================================================================

"""
scenario.md is used by SOGARL.

However, the complete scenario.md MUST NOT be passed to Red or Blue because
it can contain information that reveals the vulnerability, attack, root cause,
or defense.

Only the sections listed below are allowed to become model-visible scenario
context.

IMPORTANT:

This is contextual information only. It is NOT ground truth.
Ground-truth vulnerability information remains inside metadata.json and is
used by the Oracle.
"""

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
    "Canonical Security Boundary"}


# ============================================================================
# SFT LoRA ROOT
# ============================================================================

LORA_ROOT = Path(os.getenv("SOGARL_LORA_PATH",str(SFT_ROOT / "LoRa")))

# ============================================================================
# RED LoRA ADAPTER
# ============================================================================

"""
Expected structure:

SFT/
└── LoRa/
    └── Red_lora/
        └── Llama/
            └── results/
                └── outputs/
                    └── lora_adapter/
"""

RED_ADAPTER_PATH = Path(os.getenv("SOGARL_RED_ADAPTER",str(
            LORA_ROOT
            / "Red_lora"
            / "Llama"
            / "results"
            / "outputs"
            / "lora_adapter")))


# ============================================================================
# BLUE LoRA ADAPTER
# ============================================================================

"""
Expected structure:

SFT/
└── LoRa/
    └── Blue_lora/
        └── Llama/
            └── results/
                └── outputs/
                    └── lora_adapter/
"""

BLUE_ADAPTER_PATH = Path(os.getenv("SOGARL_BLUE_ADAPTER",str(
            LORA_ROOT
            / "Blue_lora"
            / "Llama"
            / "results"
            / "outputs"
            / "lora_adapter")))


# ============================================================================
# LOCAL MERGED MODELS
# ============================================================================

"""
Merged models are optional.

The preferred SOGARL model configuration is:

    Base model:
        Hugging Face
        unsloth/Meta-Llama-3.1-8B-Instruct

    Red:
        Base model + Red LoRA adapter

    Blue:
        Base model + Blue LoRA adapter

Merged models are retained as optional configuration values for
compatibility with local development.

They are NOT required for Kaggle execution.

On Kaggle, the paths can remain unset/non-existent because the
input dataset contains the LoRA adapters, while the base model is
downloaded from Hugging Face.
"""

RED_MERGED_MODEL_PATH = Path(os.getenv(
    "SOGARL_RED_MERGED_MODEL",
    str(LORA_ROOT / "Red_lora" / "Llama" / "results" / "outputs" / "merged_model"),
)).expanduser()

BLUE_MERGED_MODEL_PATH = Path(os.getenv(
    "SOGARL_BLUE_MERGED_MODEL",
    str(LORA_ROOT / "Blue_lora" / "Llama" / "results" / "outputs" / "merged_model"),
)).expanduser()

# ============================================================================
# BASE MODEL
# ============================================================================

"""
Base model used by Red, Blue, and the Oracle fallback.

Default:

    unsloth/Meta-Llama-3.1-8B-Instruct

This is a Hugging Face repository ID.

When running on Kaggle, the model is downloaded from Hugging Face
and the corresponding Red/Blue LoRA adapter is attached.

The local merged model paths above are optional and are not required
for Kaggle.
"""

# May be a Hugging Face repo ID OR a local/Kaggle model directory.
# Priority: explicit local path, then existing override, then HF default.
_BASE_MODEL_PATH = os.getenv("SOGARL_BASE_MODEL_PATH", "").strip()
_BASE_MODEL_OVERRIDE = os.getenv("SOGARL_BASE_MODEL", "").strip()
BASE_MODEL_NAME = (
    _BASE_MODEL_PATH
    if _BASE_MODEL_PATH
    else (_BASE_MODEL_OVERRIDE or "unsloth/Meta-Llama-3.1-8B-Instruct")
)

BASE_MODEL_SOURCE = os.getenv("SOGARL_BASE_MODEL_SOURCE", "auto").strip().lower()
if BASE_MODEL_SOURCE not in {"auto", "huggingface", "local"}:
    raise ValueError(
        "SOGARL_BASE_MODEL_SOURCE must be one of: auto, huggingface, local.")

# ============================================================================
# ORACLE MODEL
# ============================================================================

"""
The Oracle is separate from the Red and Blue policies.

It evaluates generated responses against metadata.json.

By default, the Oracle uses BASE_MODEL_NAME.

If a separate Oracle model is desired, set:

    SOGARL_ORACLE_MODEL

through the environment.

The Oracle does NOT use the Red/Blue LoRA adapters.
"""

ORACLE_MODEL_NAME = os.getenv("SOGARL_ORACLE_MODEL", BASE_MODEL_NAME)
ORACLE_USE_LORA = False
ORACLE_MODEL_SOURCE = os.getenv(
    "SOGARL_ORACLE_MODEL_SOURCE", "auto"
).strip().lower()
if ORACLE_MODEL_SOURCE not in {"auto", "huggingface", "local"}:
    raise ValueError(
        "SOGARL_ORACLE_MODEL_SOURCE must be one of: auto, huggingface, local."
    )

# ============================================================================
# OUTPUT PATHS
# ============================================================================

"""
All generated SOGARL artifacts must be written to a writable location.

Kaggle:

    /kaggle/input/   -> READ ONLY
    /kaggle/working/ -> WRITABLE

Therefore SOGARL_OUTPUT_DIR can override the default output directory.

Local default:

    SOGARL_ROOT / outputs

Kaggle:

    /kaggle/working/SOGARL/outputs
"""

_DEFAULT_OUTPUT_ROOT = (
    Path("/kaggle/working/SOGARL/outputs")
    if SOGARL_RUNTIME == "kaggle"
    else SOGARL_ROOT / "outputs"
)

OUTPUTS_PATH = Path(
    os.getenv("SOGARL_OUTPUT_DIR", str(_DEFAULT_OUTPUT_ROOT))
).expanduser()

CHECKPOINT_PATH = (
    OUTPUTS_PATH / "checkpoints"
)

REPLAY_PATH = (
    OUTPUTS_PATH / "replay"
)

METRICS_PATH = (
    OUTPUTS_PATH / "metrics"
)

LOG_PATH = (
    OUTPUTS_PATH / "logs"
)


# ============================================================================
# DATASET SPLIT
# ============================================================================

"""
Scenarios are split at the scenario level.

A complete scenario belongs to exactly one split.

We do NOT split individual candidate responses.
"""

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15

SPLIT_SEED = 42

MIN_SCENARIOS = 1


# ============================================================================
# TRAINING
# ============================================================================

SEED = 42

NUM_EPOCHS = 3

# None means:
# process every training scenario once per epoch.

MAX_EPISODES = None

SHUFFLE_SCENARIOS = True


# ============================================================================
# LOCKED SOGARL EPISODE STRUCTURE
# ============================================================================

"""
TURN 1 — RED ATTACK DISCOVERY
----------------------------

Repository
+
Safe Scenario Context

        ↓

Red LoRA

        ↓

G attack candidates

        ↓

Oracle scores all G candidates

        ↓

Red Turn-1 Oracle advantages


TURN 2 — BLUE DEFENSE GENERATION
--------------------------------

Repository
+
Safe Scenario Context

        +

Attack_best
        OR
Independent fallback

        ↓

Blue LoRA

        ↓

G defense candidates

        ↓

Oracle scores all G candidates

        ↓

Blue Oracle advantages

        ↓

Top-K defenses selected


TURN 3 — RED ADVERSARIAL CHALLENGE
-----------------------------------

For each Top-K Blue defense:

Repository
+
Safe Scenario Context
+
Blue defense

        ↓

Red LoRA

        ↓

Red challenge response

        ↓

Oracle determines interaction outcome


IMPORTANT:

Turn 3 is interaction evidence in v1.

It does NOT create another Red GRPO update.

The interaction reward is attributed to the
Blue defense candidate that was actually challenged.
"""


# ============================================================================
# GRPO ROLLOUTS
# ============================================================================

NUM_ROLLOUTS = 8

TOP_K = 3


# ============================================================================
# GENERATION
# ============================================================================

RED_TEMPERATURE = 0.8

BLUE_TEMPERATURE = 0.8

CHALLENGE_TEMPERATURE = 0.7

TOP_P = 0.95

DO_SAMPLE = True

MAX_INPUT_TOKENS = 6148

MAX_NEW_TOKENS = 1024

GENERATION_BATCH_SIZE = 1


# ============================================================================
# GRPO OPTIMIZATION
# ============================================================================

LEARNING_RATE_RED = 1e-5
LEARNING_RATE_BLUE = 1e-5
WEIGHT_DECAY = 0.01
GRPO_CLIP_EPSILON = 0.2
KL_COEFFICIENT = 0.02
ADVANTAGE_EPSILON = 1e-8
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
USE_MERGED_MODEL = False


# ============================================================================
# REWARD FUSION
# ============================================================================

"""
Only Blue receives interaction advantage in v1.

For Blue candidate i:

    A_final_i =
        alpha * A_oracle_i
        +
        beta * A_interaction_i


A_oracle
--------

Computed over all G Blue candidates.


A_interaction
-------------

Computed only over the Top-K Blue candidates that
were actually challenged by Red.

Candidates outside Top-K receive:

    A_interaction = 0


Red Turn 1
----------

Red receives:

    A_final = A_oracle


Red Turn 3
----------

No GRPO update in v1.

The Turn-3 response is used as interaction evidence
and evaluation data.
"""

ALPHA_MAX = 1.0

BETA_MAX = 0.20

BETA_START_MEAN = 0.60

BETA_FULL_MEAN = 0.75

USE_CONTINUOUS_BETA = True


# ============================================================================
# CONFIDENCE GATE
# ============================================================================

"""
The confidence gate controls ONLY whether Blue receives
Red's Turn-1 Attack_best as context.

It does NOT disable Turn 3.

If Red is sufficiently reliable for the current category:

    Blue sees:

        Repository
        +
        Safe Scenario Context
        +
        Attack_best

Otherwise:

    Blue receives an independent defense prompt:

        Repository
        +
        Safe Scenario Context

Turn 3 always happens after Blue generates and ranks
its defense candidates.
"""

USE_CONFIDENCE_GATE = True

CONFIDENCE_STD_MULTIPLIER = 0.5

MIN_CATEGORY_EPISODES_FOR_GATE = 20

BLUE_USE_INDEPENDENT_FALLBACK = True


# ============================================================================
# INTERACTION REWARD MATRIX
# ============================================================================

"""
CASE 1
------

Red claims:
    Correct attack X exists.

Ground truth:
    X remains exploitable.

Reward:
    Red  = +1
    Blue = -1


CASE 2
------

Red claims:
    Fabricated attack X.

Ground truth:
    No vulnerability remains.

Reward:
    Red  = -1
    Blue = +1


CASE 3
------

Red claims:
    Wrong attack X.

Ground truth:
    Different real vulnerability Y remains.

Reward:
    Red  = 0
    Blue = -1


CASE 4
------

Red claims:
    No attack remains.

Ground truth:
    Defense genuinely closes vulnerability.

Reward:
    Red  = +1
    Blue = +1


CASE 5
------

Red claims:
    No attack remains.

Ground truth:
    A real vulnerability remains.

Reward:
    Red  = -1
    Blue = -1
"""


INTERACTION_REWARD_CORRECT_ATTACK = 1.0

INTERACTION_REWARD_WRONG_ATTACK_NO_REMAINING = -1.0

INTERACTION_REWARD_WRONG_ATTACK_REAL_REMAINING = 0.0

INTERACTION_REWARD_NO_ATTACK_CORRECT = 1.0

INTERACTION_REWARD_NO_ATTACK_MISSED = -1.0


BLUE_REWARD_VALID_ATTACK_REMAINS = -1.0

BLUE_REWARD_NO_ATTACK_REMAINS = 1.0


# ============================================================================
# INTERACTION ADVANTAGE NORMALIZATION
# ============================================================================

NORMALIZE_INTERACTION_ADVANTAGE = True

INTERACTION_STD_EPSILON = 1e-8


# ============================================================================
# ORACLE
# ============================================================================

"""
The Oracle uses two types of evaluation.

1. Deterministic checks
-----------------------

Used wherever possible.

Examples:

    - attack label
    - relevant files
    - format
    - structured fields


2. Semantic judging
-------------------

Used only where deterministic comparison is insufficient.

Examples:

    - reasoning quality
    - semantic correctness
    - hallucination / explanation quality

This reduces the reward-hacking surface.
"""

USE_DETERMINISTIC_CHECKS = True

USE_SEMANTIC_ORACLE = True

ORACLE_JUDGMENTS = 3

ORACLE_TEMPERATURE = 0.2


# ============================================================================
# ORACLE SCORE WEIGHTS
# ============================================================================

"""
Total Oracle reward:

    Attack label       = 0.30
    Root cause         = 0.20
    Relevant files     = 0.20
    Reasoning quality  = 0.20
    Format             = 0.10

Total:

    1.00
"""

ATTACK_LABEL_WEIGHT = 0.30

ROOT_CAUSE_WEIGHT = 0.20

RELEVANT_FILES_WEIGHT = 0.20

REASONING_WEIGHT = 0.20

FORMAT_WEIGHT = 0.10


# ============================================================================
# WEAKNESS MINING / SAMPLING
# ============================================================================

"""
Weakness mining affects ONLY scenario sampling.

It does NOT modify:

    - Oracle reward
    - GRPO mathematics
    - metadata.json
    - scenario files
    - candidate rewards

Categories with sufficiently strong evidence of poor
performance receive increased sampling probability.

Sampling:

    70% → weak categories
    30% → random categories

A category must have at least 20 episodes before it
can be considered for weakness-based prioritization.
"""

USE_WEAKNESS_SAMPLING = True

WEAK_CATEGORY_PROBABILITY = 0.70

RANDOM_CATEGORY_PROBABILITY = 0.30

WEAKNESS_MIN_EPISODES = 20

WEAKNESS_ROLLING_WINDOW = 100

WEAKNESS_USE_FAILURE_RATE = True


# ============================================================================
# TRAIN / VALIDATION / TEST BEHAVIOR
# ============================================================================

"""
TRAIN
-----

Generate candidates.
Calculate rewards.
Calculate advantages.
Perform GRPO updates.


VALIDATION
----------

Generate candidates.
Calculate Oracle rewards.
Calculate interaction metrics.

NO gradient updates.


TEST
----

Generate candidates.
Calculate Oracle rewards.
Calculate final evaluation metrics.

NO gradient updates.

Test scenarios NEVER influence training.
"""

TRAIN_UPDATES_ENABLED = True

VALIDATION_UPDATES_ENABLED = False

TEST_UPDATES_ENABLED = False


# ============================================================================
# TESTING / EVALUATION
# ============================================================================

"""
Testing evaluates the final trained Red and Blue policies
on held-out scenarios.

Test scenarios:

    - never participate in GRPO training
    - never affect weakness sampling
    - never affect beta scheduling
    - never affect confidence statistics

The final test can evaluate:

    Red attack discovery
    Blue defense generation
    Red challenge accuracy
    interaction outcomes
    Oracle reward
    category-wise performance
"""

TEST_GENERATE_INTERACTION = True

TEST_SAVE_RESPONSES = True

TEST_SAVE_ORACLE_SCORES = True


# ============================================================================
# CHECKPOINTING
# ============================================================================

SAVE_EVERY_EPOCH = True

SAVE_EVERY_N_EPISODES = 100

SAVE_BEST_CHECKPOINT = True

BEST_CHECKPOINT_METRIC = (
    "combined_validation_reward"
)

SAVE_OPTIMIZER_STATE = True

SAVE_TRAINER_STATE = True


# ============================================================================
# HARDWARE
# ============================================================================

DEVICE = os.getenv(
    "SOGARL_DEVICE",
    "cuda",
)

USE_BF16 = True

USE_FP16 = False

USE_GRADIENT_CHECKPOINTING = True

USE_KV_CACHE_DURING_TRAINING = False


# ============================================================================
# MEMORY OPTIMIZATION
# ============================================================================

"""
These options are intentionally conservative.

They can be enabled later if the selected GPU requires
additional memory reduction.
"""

USE_8BIT_OPTIMIZER = False

USE_4BIT_MODEL = True

MAX_MEMORY_GB = None


# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"

SAVE_EPISODE_JSONL = True

SAVE_TRAINING_METRICS = True

SAVE_CATEGORY_METRICS = True

PRINT_EPISODE_SUMMARY = True

LOG_EVERY_EPISODES = 1


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

DETERMINISTIC_MODE = False

# Runtime values consumed directly by RewardManager.
ALPHA = 0.8
ORACLE_ADVANTAGE_EPSILON = 1e-8
CATEGORY_REWARD_HISTORY_SIZE = 100
REPLAY_BUFFER_CAPACITY = 5000


# ============================================================================
# CONFIG VALIDATION
# ============================================================================

def validate_config():
    """
    Validate configuration before training starts.
    """

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset path does not exist:\n"
            f"{DATASET_PATH}"
        )

    # ------------------------------------------------------------------------
    # LoRA adapters
    # ------------------------------------------------------------------------

    if not RED_ADAPTER_PATH.exists():

        raise FileNotFoundError(
            f"Red adapter path does not exist:\n"
            f"{RED_ADAPTER_PATH}"
        )

    if not BLUE_ADAPTER_PATH.exists():

        raise FileNotFoundError(
            f"Blue adapter path does not exist:\n"
            f"{BLUE_ADAPTER_PATH}"
        )

    # ------------------------------------------------------------------------
    # Base model
    # ------------------------------------------------------------------------

    if not BASE_MODEL_NAME:

        raise ValueError(
            "BASE_MODEL_NAME cannot be empty."
        )

    # ------------------------------------------------------------------------
    # Merged models
    #
    # Merged models are optional because generator.py has fallback behavior.
    # ------------------------------------------------------------------------

    if (
        RED_MERGED_MODEL_PATH.exists()
        and not RED_MERGED_MODEL_PATH.is_dir()
    ):

        raise ValueError(
            "RED_MERGED_MODEL_PATH exists but "
            "is not a directory:\n"
            f"{RED_MERGED_MODEL_PATH}"
        )

    if (
        BLUE_MERGED_MODEL_PATH.exists()
        and not BLUE_MERGED_MODEL_PATH.is_dir()
    ):

        raise ValueError(
            "BLUE_MERGED_MODEL_PATH exists but "
            "is not a directory:\n"
            f"{BLUE_MERGED_MODEL_PATH}"
        )

    # ------------------------------------------------------------------------
    # Dataset split
    # ------------------------------------------------------------------------

    split_total = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(
        split_total - 1.0
    ) > 1e-6:

        raise ValueError(
            "TRAIN_RATIO + VALIDATION_RATIO + "
            "TEST_RATIO must equal 1.0."
        )

    # ------------------------------------------------------------------------
    # GRPO
    # ------------------------------------------------------------------------

    if NUM_ROLLOUTS < 2:

        raise ValueError(
            "NUM_ROLLOUTS must be at least 2."
        )

    if not (
        1
        <= TOP_K
        <= NUM_ROLLOUTS
    ):

        raise ValueError(
            "TOP_K must be between "
            "1 and NUM_ROLLOUTS."
        )

    # ------------------------------------------------------------------------
    # Beta
    # ------------------------------------------------------------------------

    if not (
        0
        <= BETA_START_MEAN
        <= BETA_FULL_MEAN
        <= 1
    ):

        raise ValueError(
            "Invalid beta thresholds."
        )

    if BETA_MAX < 0:

        raise ValueError(
            "BETA_MAX cannot be negative."
        )

    # ------------------------------------------------------------------------
    # Weakness sampling
    # ------------------------------------------------------------------------

    sampling_total = (
        WEAK_CATEGORY_PROBABILITY
        + RANDOM_CATEGORY_PROBABILITY
    )

    if abs(
        sampling_total - 1.0
    ) > 1e-6:

        raise ValueError(
            "Weakness sampling probabilities "
            "must sum to 1.0."
        )

    # ------------------------------------------------------------------------
    # Oracle weights
    # ------------------------------------------------------------------------

    oracle_weight_total = (
        ATTACK_LABEL_WEIGHT
        + ROOT_CAUSE_WEIGHT
        + RELEVANT_FILES_WEIGHT
        + REASONING_WEIGHT
        + FORMAT_WEIGHT
    )

    if abs(
        oracle_weight_total - 1.0
    ) > 1e-6:

        raise ValueError(
            "Oracle score weights must sum to 1.0."
        )

    # ------------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------------

    if MAX_INPUT_TOKENS <= 0:

        raise ValueError(
            "MAX_INPUT_TOKENS must be positive."
        )

    if MAX_NEW_TOKENS <= 0:

        raise ValueError(
            "MAX_NEW_TOKENS must be positive."
        )

    # ------------------------------------------------------------------------
    # Runtime / paths
    # ------------------------------------------------------------------------

    if BASE_MODEL_SOURCE == "local" or Path(BASE_MODEL_NAME).is_absolute():
        base_path = Path(BASE_MODEL_NAME).expanduser()
        if not base_path.is_dir():
            raise FileNotFoundError(
                f"Local BASE_MODEL_NAME does not exist or is not a directory:\n{base_path}"
            )

    if ORACLE_MODEL_SOURCE == "local" or Path(ORACLE_MODEL_NAME).is_absolute():
        oracle_path = Path(ORACLE_MODEL_NAME).expanduser()
        if not oracle_path.is_dir():
            raise FileNotFoundError(
                f"Local ORACLE_MODEL_NAME does not exist or is not a directory:\n{oracle_path}"
            )

    if CONFIDENCE_STD_MULTIPLIER < 0:
        raise ValueError("CONFIDENCE_STD_MULTIPLIER cannot be negative.")

    if MIN_CATEGORY_EPISODES_FOR_GATE < 1:
        raise ValueError("MIN_CATEGORY_EPISODES_FOR_GATE must be positive.")

    if GRADIENT_ACCUMULATION_STEPS < 1:
        raise ValueError("GRADIENT_ACCUMULATION_STEPS must be positive.")

    if GENERATION_BATCH_SIZE < 1:
        raise ValueError("GENERATION_BATCH_SIZE must be positive.")

    if USE_BF16 and USE_FP16:
        raise ValueError("USE_BF16 and USE_FP16 cannot both be True.")

    if DEVICE not in {"cuda", "cpu", "auto"}:
        raise ValueError("DEVICE must be one of: cuda, cpu, auto.")

    # ------------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------------

    if TOP_K > NUM_ROLLOUTS:

        raise ValueError(
            "TOP_K cannot exceed NUM_ROLLOUTS."
        )


# ============================================================================
# DEAD / LEGACY SETTINGS (KEPT; NAMES MUST NOT BE REMOVED)
# ============================================================================
# The audited codebase currently does not consume these settings at runtime:
#
# ORACLE_USE_LORA, MIN_SCENARIOS, SHUFFLE_SCENARIOS,
# WEAKNESS_USE_FAILURE_RATE, VALIDATION_UPDATES_ENABLED, TEST_UPDATES_ENABLED,
# TEST_GENERATE_INTERACTION, TEST_SAVE_RESPONSES, TEST_SAVE_ORACLE_SCORES,
# BEST_CHECKPOINT_METRIC, SAVE_OPTIMIZER_STATE, SAVE_TRAINER_STATE,
# USE_KV_CACHE_DURING_TRAINING, USE_8BIT_OPTIMIZER, MAX_MEMORY_GB,
# SAVE_EPISODE_JSONL, SAVE_TRAINING_METRICS, SAVE_CATEGORY_METRICS,
# PRINT_EPISODE_SUMMARY, DETERMINISTIC_MODE.
#
# They are retained for compatibility and documentation; changing them does
# not currently change the corresponding runtime behavior.
#
# ============================================================================

# ============================================================================
# OUTPUT DIRECTORY CREATION
# ============================================================================

def create_output_directories():
    """
    Create writable SOGARL output directories.
    """

    paths = (
        OUTPUTS_PATH,
        CHECKPOINT_PATH,
        REPLAY_PATH,
        METRICS_PATH,
        LOG_PATH,
    )

    for path in paths:

        path.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================================
# CONFIGURATION DISPLAY
# ============================================================================

def print_config():
    """
    Print important runtime configuration.
    """

    print("=" * 70)

    print(
        "SOGARL CONFIGURATION"
    )

    print("=" * 70)

    print(
        f"SOGARL ROOT : {SOGARL_ROOT}"
    )

    print(
        f"Dataset     : {DATASET_PATH}"
    )

    print(
        f"Red LoRA    : {RED_ADAPTER_PATH}"
    )

    print(
        f"Blue LoRA   : {BLUE_ADAPTER_PATH}"
    )

    print(
        f"Red merged  : {RED_MERGED_MODEL_PATH}"
    )

    print(
        f"Blue merged : {BLUE_MERGED_MODEL_PATH}"
    )

    print(
        f"Base model  : {BASE_MODEL_NAME}"
    )

    print(
        f"Oracle      : {ORACLE_MODEL_NAME}"
    )

    print()

    print(
        f"G           : {NUM_ROLLOUTS}"
    )

    print(
        f"Top-K       : {TOP_K}"
    )

    print(
        f"Epochs      : {NUM_EPOCHS}"
    )

    print()

    print(
        f"Red LR      : {LEARNING_RATE_RED}"
    )

    print(
        f"Blue LR     : {LEARNING_RATE_BLUE}"
    )

    print(
        f"KL          : {KL_COEFFICIENT}"
    )

    print(
        f"Beta max    : {BETA_MAX}"
    )

    print()

    print(
        f"Confidence  : {USE_CONFIDENCE_GATE}"
    )

    print(
        f"Weakness    : {USE_WEAKNESS_SAMPLING}"
    )

    print(
        f"Context     : {MAX_INPUT_TOKENS}"
    )

    print(
        f"Device      : {DEVICE}"
    )

    print("=" * 70)