# SOGARL — Final Implementation File Map

## 1. Final Project Structure

```text
SOGARL/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.py
│
├── src/
│   │
│   ├── data/
│   │   ├── models.py
│   │   ├── scenario_loader.py
│   │   ├── metadata_loader.py
│   │   ├── path_resolver.py
│   │   └── code_loader.py
│   │
│   ├── generation/
│   │   ├── prompt_builder.py
│   │   └── generator.py
│   │
│   ├── oracle/
│   │   ├── oracle.py
│   │   ├── deterministic_checks.py
│   │   └── interaction_checker.py
│   │
│   ├── rl/
│   │   ├── grpo.py
│   │   ├── reward_manager.py
│   │   └── episode_manager.py
│   │
│   ├── sampling/
│   │   └── weakness_sampler.py
│   │
│   ├── logging/
│   │   ├── replay_buffer.py
│   │   └── metrics_logger.py
│   │
│   └── utils/
│       ├── file_utils.py
│       ├── random_utils.py
│       ├── context_estimator.py
│       ├── logger.py
│       ├── checkpoint.py
│       ├── seed.py
│       └── device.py
│
├── scripts/
│   ├── validate_dataset.py
│   ├── test_oracle.py
│   ├── test_episode.py
│   ├── train.py
│   └── evaluate.py
│
├── outputs/
│   ├── checkpoints/
│   ├── replay/
│   │   └── episodes.jsonl
│   ├── metrics/
│   │   ├── training_metrics.jsonl
│   │   ├── red_metrics.csv
│   │   ├── blue_metrics.csv
│   │   └── weakness_report.json
│   └── logs/
│       └── training.log
│
└── notebooks/
    ├── 01_validate_dataset.ipynb
    ├── 02_test_oracle.ipynb
    └── 03_debug_episode.ipynb
```

---

# 2. Root Files

## `README.md`

Project documentation.

It explains:

- What SOGARL is
- Red and Blue agents
- Three-turn RL pipeline
- Oracle architecture
- GRPO training
- Dataset requirements
- Installation
- Testing
- Training
- Evaluation
- Checkpoint/resume procedure
- Kaggle/VCloud execution

It does not participate directly in training.

---

## `requirements.txt`

Contains all Python dependencies required by SOGARL.

Typical dependencies include:

```text
torch
transformers
peft
accelerate
datasets
numpy
pandas
scipy
```

Additional dependencies should be added only when actually required by the implementation.

---

## `.gitignore`

Prevents generated or large files from being committed.

Examples:

```text
outputs/
checkpoints/
*.safetensors
*.pt
*.pth
__pycache__/
*.log
```

---

# 3. Configuration

## `configs/config.py`

This is the **single central configuration file**.

It contains:

- Dataset paths
- Red LoRA path
- Blue LoRA path
- Base model path/name
- Group size `G`
- Top-K value `K`
- Learning rates
- Batch sizes
- Gradient accumulation
- Epoch count
- Confidence-gate settings
- `alpha` / `beta`
- Weakness-sampling settings
- Checkpoint frequency
- Logging frequency
- Train/validation/test scenario split
- Device and precision settings

Example conceptual settings:

```python
GROUP_SIZE = 8
TOP_K = 3

RED_LORA_PATH = ...
BLUE_LORA_PATH = ...

TRAIN_SCENARIOS = [...]
VAL_SCENARIOS = [...]
TEST_SCENARIOS = [...]
```

No other module should hardcode environment-specific paths.

---

# 4. Data Layer

## `src/data/models.py`

Contains the structured data objects used throughout SOGARL.

Examples:

```text
Scenario
ScenarioMetadata
AttackCandidate
DefenseCandidate
OracleResult
InteractionResult
EpisodeResult
```

Instead of passing unstructured dictionaries between modules, the system passes well-defined objects.

This keeps the architecture decoupled.

---

## `src/data/scenario_loader.py`

Loads an individual scenario.

Example:

```text
scenario_021/
├── scenario.md
├── metadata.json
└── codebase/
```

It creates a `Scenario` object containing the information required by the rest of the pipeline.

It does not:

- Generate prompts
- Calculate rewards
- Run Oracle logic
- Load SFT examples for RL

---

## `src/data/metadata_loader.py`

Loads and validates:

```text
metadata.json
```

The metadata contains hidden ground truth used by the Oracle.

Depending on the existing schema, this can include:

- Valid attacks
- Valid defenses
- Vulnerability information
- Relevant files
- Root-cause information
- Other ground-truth fields

Important information boundary:

```text
Model → NEVER receives hidden metadata
Oracle → CAN access hidden metadata
```

---

## `src/data/path_resolver.py`

Provides environment-independent path handling.

The same code should work with paths such as:

```text
Windows:
F:\SEM-7\Capstone\...

Kaggle:
/kaggle/input/...

VCloud:
/workspace/...
```

The rest of the project should not need to know which environment it is running on.

---

## `src/data/code_loader.py`

Loads the source-code repository contained inside a scenario's `codebase/` directory.

Example:

```text
codebase/
├── auth.py
├── routes.py
├── database.py
└── utils.py
```

It converts the files into repository context that can be supplied to the models.

It does not evaluate correctness.

---

# 5. Generation Layer

## `src/generation/prompt_builder.py`

Builds the prompts for the three SOGARL turns.

### Red Turn 1

```text
Repository
+
Scenario
```

Task:

```text
Find the vulnerability / attack.
```

### Blue Turn 2

If Red passes the confidence gate:

```text
Repository
+
Scenario
+
Red's best attack
```

Otherwise:

```text
Repository
+
Scenario
```

### Red Turn 3

```text
Repository
+
Scenario
+
Blue defense
```

Task:

```text
Determine whether a valid attack remains.
```

This file controls the information flow between agents.

---

## `src/generation/generator.py`

Actually performs model generation.

Responsibilities include:

- Tokenization
- Generation
- Sampling
- Temperature
- Top-p
- Maximum generation length
- Batching
- Attention masks
- Device placement

Conceptual interfaces:

```python
generate_red_attacks(...)
generate_blue_defenses(...)
generate_red_challenge(...)
```

The generator does not determine whether a response is correct.

---

# 6. Oracle Layer

## `src/oracle/deterministic_checks.py`

Performs objective checks that do not require an LLM.

Examples:

- Valid attack label
- Valid defense label
- Relevant-file overlap
- Required fields
- Output structure
- Fabricated attack detection
- Ground-truth label matching

These checks are deterministic.

---

## `src/oracle/oracle.py`

Main Oracle coordinator.

It combines:

```text
Deterministic checks
        +
Semantic evaluation
        ↓
Oracle reward
```

For Red, it can evaluate:

- Attack correctness
- Relevant files
- Root cause
- Reasoning quality
- Hallucination

For Blue:

- Defense correctness
- Relevant files
- Mitigation quality
- Reasoning quality

The Oracle should return a structured result, not only a scalar.

Conceptually:

```text
OracleResult
├── total_reward
├── correctness_score
├── file_score
├── reasoning_score
├── hallucination_penalty
└── details
```

The Oracle is not trained during normal SOGARL training.

---

## `src/oracle/interaction_checker.py`

Handles Turn-3 interaction scoring.

The locked interaction matrix is:

| Case | Red Claim | Actual State | Red | Blue |
|---|---|---|---:|---:|
| 1 | Correct attack X remains | X remains exploitable | +1 | -1 |
| 2 | Fabricated attack X | No vulnerability remains | -1 | +1 |
| 3 | Wrong attack X | Different real attack Y remains | 0 | -1 |
| 4 | No attack remains | No vulnerability remains | +1 | +1 |
| 5 | No attack remains | Real vulnerability remains | -1 | -1 |

Critical rule:

> Blue's interaction reward depends on the actual security state of the defense, not on whether Red successfully identifies the remaining vulnerability.

Therefore:

```text
Bad Blue defense
+
Red misses the vulnerability
=
Blue still receives -1
```

This prevents Red's weakness from rescuing Blue.

---

# 7. RL Layer

## `src/rl/reward_manager.py`

Responsible for reward and advantage mathematics.

It receives:

```text
Oracle rewards
Interaction rewards
```

and produces:

```text
Oracle advantages
Interaction advantages
Final advantages
```

Oracle advantage:

```text
A_oracle =
    (R_oracle - mean(R_oracle))
    /
    (std(R_oracle) + epsilon)
```

Interaction advantage:

```text
A_interaction =
    (R_interaction - mean(R_interaction))
    /
    (std(R_interaction) + epsilon)
```

Final advantage:

```text
A_final =
    alpha * A_oracle
    +
    beta * A_interaction
```

For candidates that were not interaction-tested:

```text
A_interaction = 0
```

Reward fusion happens at the **advantage level**, not by adding raw rewards.

---

## `src/rl/grpo.py`

Implements the actual GRPO optimization.

Conceptual flow:

```text
Responses
    ↓
Log probabilities
    ↓
Advantages
    ↓
GRPO objective
    ↓
KL regularization
    ↓
Loss
    ↓
Backpropagation
    ↓
LoRA update
```

Red and Blue are optimized independently.

One episode produces:

```text
1 Red GRPO update
+
1 Blue GRPO update
```

Turn 3 does not create a second Red GRPO update in V1.

---

## `src/rl/episode_manager.py`

This is the **central SOGARL episode orchestrator**.

It coordinates:

```text
Scenario
   ↓
Red Turn 1
   ↓
Oracle
   ↓
Confidence Gate
   ↓
Blue Turn 2
   ↓
Oracle
   ↓
Top-K selection
   ↓
Red Turn 3 challenges
   ↓
Interaction Checker
   ↓
Reward Manager
   ↓
Red GRPO
   ↓
Blue GRPO
   ↓
Replay Buffer
   ↓
Metrics
```

The episode manager should coordinate modules rather than duplicate their internal logic.

---

# 8. Sampling Layer

## `src/sampling/weakness_sampler.py`

Controls which scenario is selected next.

It uses historical performance such as:

- Failure rate
- Oracle reward
- Interaction failures
- Vulnerability category
- Number of observations

Example:

```text
70% → weak categories
30% → random categories
```

A minimum observation threshold should be used before declaring a category weak.

Weakness sampling changes the training distribution.

It does **not** modify the reward.

---

# 9. Logging Layer

## `src/logging/replay_buffer.py`

Stores completed episode records.

A record can contain:

```text
Episode ID
Scenario ID
Category

Red candidates
Red Oracle rewards
Red advantages

Blue candidates
Blue Oracle rewards
Blue advantages

Top-K interaction results
Interaction advantages

Final advantages
Episode outcome
```

The replay buffer supports:

- Weakness mining
- Debugging
- Analysis
- Visualization
- Training diagnostics

---

## `src/logging/metrics_logger.py`

Tracks training statistics.

Examples:

```text
Red Oracle reward
Blue Oracle reward

Red loss
Blue loss

KL divergence

Interaction reward

Confidence-gate pass rate

Hallucination rate

Defense survival rate

Category failure rate
```

Outputs include:

```text
training_metrics.jsonl
red_metrics.csv
blue_metrics.csv
```

---

# 10. Utility Layer

## `src/utils/file_utils.py`

Generic filesystem operations:

- Read files
- Write files
- Read JSON
- Write JSON
- List directories
- Create directories
- Check file existence

This avoids duplicating filesystem logic throughout the project.

---

## `src/utils/random_utils.py`

Centralizes random operations:

- Sampling
- Shuffling
- Random scenario selection
- Candidate selection

Works together with the global seed.

---

## `src/utils/context_estimator.py`

Estimates repository prompt size before generation.

This is particularly important for Kaggle/VCloud because large repositories can cause:

- Context-length errors
- GPU OOM
- Excessive generation time

Conceptually:

```text
Repository
    ↓
Context Estimator
    ↓
Fits context?
   / \
 Yes  No
 |     |
Send   Controlled truncation
```

The truncation policy should preserve the most relevant repository information rather than blindly cutting characters.

---

## `src/utils/logger.py`

Central logging setup.

Instead of scattered `print()` statements:

```python
logger.info(...)
logger.warning(...)
logger.error(...)
```

Logs are written to:

```text
outputs/logs/training.log
```

---

## `src/utils/checkpoint.py`

Handles checkpoint creation and restoration.

A checkpoint should preserve:

```text
Red LoRA
Blue LoRA

Red optimizer state
Blue optimizer state

Current epoch
Current episode

Scheduler state

Random-number-generator state

Metrics

Training configuration
```

This is essential because Kaggle has a runtime limit.

If a runtime ends:

```text
Load checkpoint
      ↓
Restore state
      ↓
Continue training
```

---

## `src/utils/seed.py`

Controls reproducibility across:

```text
Python
NumPy
PyTorch
CUDA
Generation
```

---

## `src/utils/device.py`

Handles:

```text
CUDA
CPU
GPU selection
Mixed precision
Device placement
```

The same implementation can therefore be used on:

```text
Kaggle
VCloud
Local machine
```

---

# 11. Executable Scripts

The `src/` directory contains implementation modules.

The `scripts/` directory contains the files you actually execute.

---

## `scripts/validate_dataset.py`

First execution step.

Checks:

```text
Scenario count
Scenario directories
metadata.json
scenario.md
codebase/
Required metadata fields
Valid scenario structure
```

Run:

```bash
python scripts/validate_dataset.py
```

No training occurs.

---

## `scripts/test_oracle.py`

Tests the Oracle before RL training.

It must explicitly test the five interaction cases:

```text
Case 1 → Red +1, Blue -1
Case 2 → Red -1, Blue +1
Case 3 → Red 0, Blue -1
Case 4 → Red +1, Blue +1
Case 5 → Red -1, Blue -1
```

If Oracle tests fail:

```text
STOP
```

Training should not begin.

Run:

```bash
python scripts/test_oracle.py
```

---

## `scripts/test_episode.py`

Runs one complete SOGARL episode for integration testing.

It should display:

```text
Red candidates
Red Oracle rewards
Red advantages

Confidence gate result

Blue candidates
Blue Oracle rewards
Blue advantages

Top-K defenses

Red challenges

Interaction matrix

Interaction advantages

Final advantages
```

During initial debugging:

```text
NO optimizer update
```

This is an integration/debugging tool.

Run:

```bash
python scripts/test_episode.py
```

---

## `scripts/train.py`

This is the **actual RL training entry point**.

Run:

```bash
python scripts/train.py
```

It performs:

```text
Load base model
       ↓
Load Red LoRA
       ↓
Load Blue LoRA
       ↓
Load training scenarios
       ↓
Select scenario
       ↓
Run SOGARL episode
       ↓
Update Red
       ↓
Update Blue
       ↓
Log
       ↓
Checkpoint
       ↓
Next episode
```

You do not manually execute:

```text
grpo.py
oracle.py
episode_manager.py
```

`train.py` orchestrates them.

---

## `scripts/evaluate.py`

This is the **final research evaluation entry point**.

It runs after RL training.

It loads a frozen checkpoint and evaluates only held-out test scenarios.

During evaluation:

```text
NO gradient
NO optimizer
NO GRPO update
NO LoRA update
NO weakness sampling
```

The pipeline is:

```text
Test scenario
      ↓
Red inference
      ↓
Oracle
      ↓
Blue inference
      ↓
Oracle
      ↓
Red challenge
      ↓
Oracle
      ↓
Metrics
```

This is separate from `test_episode.py`.

`test_episode.py` asks:

> Does the implementation work?

`evaluate.py` asks:

> Did the trained model actually improve?

---

# 12. Notebooks

## `notebooks/01_validate_dataset.ipynb`

Interactive dataset inspection.

Useful for checking:

- Scenario count
- Vulnerability distribution
- Metadata
- Codebase size
- Context lengths

---

## `notebooks/02_test_oracle.ipynb`

Interactive Oracle debugging.

Allows inspection of:

```text
Prediction
Ground truth
Deterministic scores
Semantic scores
Final reward
```

---

## `notebooks/03_debug_episode.ipynb`

Interactive visualization of a complete episode:

```text
Red
 ↓
Oracle
 ↓
Confidence gate
 ↓
Blue
 ↓
Top-K
 ↓
Red challenge
 ↓
Interaction
 ↓
Rewards
 ↓
Advantages
```

These notebooks are development tools, not production training entry points.

---

# 13. Output Directory

## `outputs/checkpoints/`

Stores resumable checkpoints.

```text
checkpoints/
├── epoch_001/
├── epoch_002/
├── epoch_003/
└── ...
```

---

## `outputs/replay/episodes.jsonl`

Stores complete episode histories.

Used by:

```text
Weakness Sampler
Metrics
Debugging
Research analysis
```

---

## `outputs/metrics/training_metrics.jsonl`

Detailed training metrics.

---

## `outputs/metrics/red_metrics.csv`

Red-specific metrics.

Examples:

```text
Epoch
Oracle reward
Attack accuracy
Hallucination rate
Interaction accuracy
GRPO loss
KL divergence
```

---

## `outputs/metrics/blue_metrics.csv`

Blue-specific metrics.

Examples:

```text
Epoch
Oracle reward
Defense accuracy
Defense survival rate
GRPO loss
KL divergence
```

---

## `outputs/metrics/weakness_report.json`

Category-level weakness analysis.

Example:

```json
{
    "SSRF": {
        "episodes": 47,
        "failure_rate": 0.61
    },
    "IDOR": {
        "episodes": 52,
        "failure_rate": 0.54
    }
}
```

---

# 14. What Data Is Used During RL?

The important information boundary is:

```text
                    Scenario
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
        Model Context       Oracle Context
             │                   │
             ├── scenario.md     ├── metadata.json
             │                   ├── valid attacks
             └── codebase/       ├── valid defenses
                                 ├── vulnerabilities
                                 └── ground truth
```

During RL:

```text
scenario.md
    ↓
Model context

codebase/
    ↓
Model context

metadata.json
    ↓
Oracle only
```

The following SFT artifacts are **not loaded by the RL pipeline**:

```text
red_sft.json
blue_sft.json
```

They remain part of the SFT dataset and are not needed for RL inference.

---

# 15. Final Execution Order

The complete development and training lifecycle is:

```text
1. validate_dataset.py
          ↓
2. test_oracle.py
          ↓
3. test_episode.py
          ↓
4. Small training run
          ↓
5. Inspect checkpoint + metrics
          ↓
6. Full train.py
          ↓
7. evaluate.py
```

Commands:

```bash
python scripts/validate_dataset.py
```

```bash
python scripts/test_oracle.py
```

```bash
python scripts/test_episode.py
```

```bash
python scripts/train.py
```

After training:

```bash
python scripts/evaluate.py
```

---

# 16. Responsibility Boundaries

The architecture is deliberately decoupled.

```text
DATA
│
├── models.py
├── scenario_loader.py
├── metadata_loader.py
├── path_resolver.py
└── code_loader.py
```

```text
GENERATION
│
├── prompt_builder.py
└── generator.py
```

```text
JUDGING
│
├── deterministic_checks.py
├── oracle.py
└── interaction_checker.py
```

```text
LEARNING
│
├── reward_manager.py
├── grpo.py
└── episode_manager.py
```

```text
CURRICULUM
│
└── weakness_sampler.py
```

```text
RECORDING
│
├── replay_buffer.py
└── metrics_logger.py
```

```text
INFRASTRUCTURE
│
├── file_utils.py
├── random_utils.py
├── context_estimator.py
├── logger.py
├── checkpoint.py
├── seed.py
└── device.py
```

This separation prevents `episode_manager.py` from becoming a monolithic file containing data loading, prompt construction, Oracle logic, reward mathematics, and optimization.

---

# 17. Final SOGARL Lifecycle

```text
                 EXISTING SFT ASSETS
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          Red LoRA              Blue LoRA
              │                     │
              └──────────┬──────────┘
                         ▼
                  TRAINING SCENARIO
                         │
                         ▼
                 RED — TURN 1
                         │
                    G attacks
                         │
                         ▼
                      ORACLE
                         │
                         ▼
                 Red advantages
                         │
                         ▼
                 Confidence Gate
                         │
                         ▼
                BLUE — TURN 2
                         │
                    G defenses
                         │
                         ▼
                      ORACLE
                         │
                         ▼
                     Top-K
                         │
                         ▼
                RED — TURN 3
                         │
                   Challenges
                         │
                         ▼
                      ORACLE
                         │
                         ▼
              Interaction rewards
                         │
                         ▼
             Interaction advantages
                         │
                         ▼
                Advantage Fusion
                         │
                  ┌──────┴──────┐
                  ▼             ▼
              RED GRPO      BLUE GRPO
                  │             │
                  ▼             ▼
               Red LoRA      Blue LoRA
                  │             │
                  └──────┬──────┘
                         ▼
                  Replay Buffer
                         │
                         ▼
                 Weakness Mining
                         │
                         ▼
                  Next Scenario
                         │
                         ▼
                   Next Episode
                         │
                         ▼
                  Checkpoint
```

---

# 18. Final Rule

The implementation should preserve these boundaries:

```text
Models
    → generate

Oracle
    → judge

Reward Manager
    → normalize and fuse rewards

GRPO
    → update policies

Episode Manager
    → orchestrate

Weakness Sampler
    → choose future scenarios

Replay Buffer
    → remember what happened

Checkpoint Manager
    → preserve training state

evaluate.py
    → measure final performance
```

The final RL pipeline therefore remains:

```text
Red Attack
    ↓
Oracle
    ↓
Confidence Gate
    ↓
Blue Defense
    ↓
Oracle
    ↓
Top-K
    ↓
Red Challenge
    ↓
Interaction Oracle
    ↓
Advantage Fusion
    ↓
Red + Blue GRPO
    ↓
Checkpoint
    ↓
Weakness Mining
    ↓
Next Episode
```

The existing SFT dataset and LoRA adapters remain outside the SOGARL source tree and are referenced through `configs/config.py`. The RL pipeline consumes the scenario description and codebase for model context and uses hidden metadata only inside the Oracle.
