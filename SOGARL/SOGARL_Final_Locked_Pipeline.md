# SOGARL --- Final Locked Pipeline {#sogarl--final-locked-pipeline}

## 1. Purpose {#1-purpose}

SOGARL trains two security-reasoning LoRA adapters:

-   **Red** --- discovers vulnerabilities and attacks.
-   **Blue** --- proposes defenses.
-   **Oracle** --- provides grounded evaluation from hidden scenario
    ground truth.
-   **Turn 3** --- Red adversarially challenges selected Blue defenses.
-   **GRPO** --- updates Red and Blue using group-relative advantages.
-   **Weakness Mining** --- increases sampling of categories where
    performance is weak.

The system is designed for static repositories. It does not execute the
target repository and does not require a manually constructed attack ×
defense outcome matrix.

The core loop is:

``` text
Red discovers
    ↓
Blue defends
    ↓
Red challenges the defense
    ↓
Oracle verifies the outcome
    ↓
GRPO updates the agents
    ↓
Weakness mining selects harder scenarios
    ↓
Repeat
```

------------------------------------------------------------------------

## 2. Locked Architecture {#2-locked-architecture}

``` text
Scenario Dataset
      ↓
Scenario Loader
      ↓
Metadata Loader
      ↓
Path Resolver
      ↓
Code Loader
      ↓
Scenario Context
      ↓
Prompt Builder
      ↓
┌─────────────────────────────────────┐
│                                     │
│ Turn 1: RED                         │
│ G=8 attack candidates               │
│        ↓                            │
│ Oracle                              │
│        ↓                            │
│ Red Oracle Advantages               │
│        ↓                            │
│ Attack_best                         │
│        ↓                            │
│ Confidence Gate                     │
│                                     │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│ Turn 2: BLUE                        │
│ G=8 defense candidates              │
│        ↓                            │
│ Oracle                              │
│        ↓                            │
│ Blue Oracle Advantages              │
│        ↓                            │
│ Blue Top-K = 3                      │
│        ↓                            │
│ Turn 3: RED challenges D1/D3/D2     │
│        ↓                            │
│ Oracle verifies challenges           │
│        ↓                            │
│ Interaction Rewards                 │
│        ↓                            │
│ Interaction Advantages              │
│        ↓                            │
│ Advantage Fusion                    │
│        ↓                            │
│ Blue GRPO                           │
└─────────────────────────────────────┘

Red Turn-1 Oracle advantages
        ↓
Red GRPO

Both updates
        ↓
Replay Buffer + Metrics
        ↓
Weakness Mining
        ↓
Next Scenario
```

------------------------------------------------------------------------

# 3. Scenario Data {#3-scenario-data}

The existing SFT dataset remains the source dataset.

``` text
SFT/
└── SFT_Dataset/
    ├── scenario_001/
    │   ├── metadata.json
    │   ├── scenario.md
    │   └── codebase/
    │       ├── ...
    │
    ├── scenario_002/
    └── ...
```

SOGARL does not modify this structure.

The data layer performs:

``` text
scenario_loader.py
        ↓
safe scenario.md sections

metadata_loader.py
        ↓
metadata.json

path_resolver.py
        ↓
all relevant files
+
2–5 noise files

code_loader.py
        ↓
CodeFile objects
        ↓
ScenarioContext
```

------------------------------------------------------------------------

# 4. Ground-Truth Isolation {#4-ground-truth-isolation}

The models must not receive the fields that directly reveal the answer.

The Oracle may access:

``` text
valid_attacks
valid_defenses
optimal_attack
optimal_defense
root cause
relevant files
vulnerability information
security boundaries
```

Red and Blue receive only:

``` text
safe scenario context
+
selected source code
+
permitted previous-turn output
```

This prevents answer copying.

------------------------------------------------------------------------

# 5. scenario.md Handling {#5-scenariomd-handling}

`scenario.md` is never passed wholesale.

Only safe contextual sections are exposed:

``` python
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
    "Canonical Security Boundary",
}
```

These sections provide:

``` text
domain context
architecture
normal behavior
technology information
trust boundaries
```

Security-answer sections that would reveal the vulnerability or defense
are excluded.

------------------------------------------------------------------------

# 6. Source-Code Loading {#6-source-code-loading}

Metadata determines the relevant files.

Example:

``` text
Relevant:
    routes.py
    user_service.py
    auth.py
```

`PathResolver` loads all metadata-listed relevant files.

It then selects only a small number of noise files:

``` text
Noise:
    logger.py
    config.py
```

`CodeLoader` does not inspect programming-language extensions.

It simply reads the files selected by `PathResolver`.

Therefore the system can handle:

``` text
Python
Java
JavaScript
TypeScript
C/C++
Go
Rust
PHP
...
```

without hard-coded language filtering.

------------------------------------------------------------------------

# 7. Prompt Construction {#7-prompt-construction}

`PromptBuilder` constructs three different task prompts.

They are not the same prompt reused three times.

``` text
Turn 1 → Red Attack Discovery
Turn 2 → Blue Defense Generation
Turn 3 → Red Adversarial Challenge
```

The repository context is shared, but the role and task instruction
changes.

------------------------------------------------------------------------

# 8. Turn 1 --- Red Attack Discovery {#8-turn-1--red-attack-discovery}

Red receives:

``` text
Scenario context
+
Repository source files
```

Conceptual prompt:

``` text
SYSTEM:

You are the Red security analyst.

Analyze the provided repository and scenario and identify
a genuine security vulnerability.

Use source-code evidence.

Return:
1. Attack category
2. Root cause
3. Relevant files
4. Step-by-step reasoning
5. Why the attack is exploitable

Do not invent files, APIs, behavior, or vulnerabilities.


USER:

Scenario context:
...

Repository:
...

Task:
Identify the strongest valid attack supported by the code.
```

Red generates:

``` text
G = 8
```

candidate attacks.

Example:

``` text
A1 → IDOR
A2 → SQL Injection
A3 → XSS
A4 → Broken Authorization
A5 → JWT Forgery
A6 → CSRF
A7 → Command Injection
A8 → RCE
```

------------------------------------------------------------------------

# 9. Red Oracle Scoring {#9-red-oracle-scoring}

Every Red candidate is scored independently.

The Oracle uses deterministic checks where possible:

``` text
attack label
relevant files
output format
```

Semantic judging is reserved for:

``` text
reasoning quality
root-cause explanation
unsupported claims
hallucination
```

Conceptually:

``` text
Oracle Reward =
    attack correctness
  + root-cause correctness
  + relevant-file correctness
  + reasoning quality
  + formatting
  - hallucination penalty
```

------------------------------------------------------------------------

# 10. Red GRPO Advantages {#10-red-grpo-advantages}

For Red\'s G candidates:

``` text
A_oracle(i) =
    (R_i - mean(R))
    /
    (std(R) + epsilon)
```

High-quality attacks receive positive relative advantages.

Weak candidates receive negative advantages.

------------------------------------------------------------------------

# 11. Red Attack Selection {#11-red-attack-selection}

The highest Oracle-scoring candidate becomes:

``` text
Attack_best
```

For example:

``` text
A1 = IDOR
```

Only `Attack_best` is handed to Blue.

The other Red candidates remain part of Red\'s GRPO group.

They are not sent to Blue.

------------------------------------------------------------------------

# 12. Confidence Gate {#12-confidence-gate}

The confidence gate determines whether Blue receives Red\'s Turn-1
finding.

If reliable:

``` text
Blue context =
    Scenario
    +
    Repository
    +
    Attack_best
```

If unreliable:

``` text
Blue context =
    Scenario
    +
    Repository
```

The gate affects only the Turn-1 Red-to-Blue handoff.

Turn 3 still occurs for Blue\'s selected defenses.

------------------------------------------------------------------------

# 13. Turn 2 --- Blue Defense Generation {#13-turn-2--blue-defense-generation}

Blue receives:

``` text
Scenario
+
Repository
+
Attack_best, if the confidence gate allows it
```

Conceptual prompt:

``` text
SYSTEM:

You are the Blue security defender.

Analyze the repository and the proposed security finding.

Design a concrete defense that removes the vulnerability
without breaking intended application behavior.

Use source-code evidence.

Return:
1. Defense strategy
2. Files that must change
3. Specific implementation logic
4. Why the defense closes the root cause
5. Security limitations or trade-offs

Do not claim that a defense works without evidence.


USER:

Scenario context:
...

Repository:
...

Red security finding:
...

Task:
Propose the strongest practical defense.
```

Blue generates:

``` text
G = 8
```

defense candidates.

Example:

``` text
D1 → Verify resource ownership
D2 → Add rate limiting
D3 → Rotate JWT secret
D4 → Input validation
D5 → CSRF protection
D6 → Firewall rule
D7 → Output sanitization
D8 → MFA
```

------------------------------------------------------------------------

# 14. Blue Oracle Scoring {#14-blue-oracle-scoring}

Every Blue candidate receives a normal Oracle reward.

Example:

``` text
D1 → 0.96
D2 → 0.21
D3 → 0.32
D4 → 0.18
D5 → 0.10
D6 → 0.07
D7 → 0.12
D8 → 0.16
```

These become:

``` text
A_oracle_blue
```

------------------------------------------------------------------------

# 15. Blue Top-K {#15-blue-top-k}

The highest K Blue defenses are selected for adversarial verification.

Locked initial value:

``` text
K = 3
```

Example:

``` text
D1
D3
D2
```

The interaction signal belongs to these Blue defenses.

It is never attached to Red\'s original attack candidates.

------------------------------------------------------------------------

# 16. Turn 3 --- Red Adversarial Challenge {#16-turn-3--red-adversarial-challenge}

Turn 3 is a different Red task.

For each selected Blue defense, Red receives:

``` text
Scenario
+
Repository
+
One Blue defense
```

Conceptual prompt:

``` text
SYSTEM:

You are the Red adversarial security reviewer.

A Blue defense has been proposed for the repository.

Determine whether a genuine exploitable vulnerability
still remains after applying the proposed defense.

If a vulnerability remains:
    identify the attack category,
    identify relevant files,
    explain how the attack bypasses the defense.

If no vulnerability remains:
    explicitly state that no valid attack remains
    and explain why the defense closes the root cause.

Do not invent vulnerabilities.


USER:

Scenario context:
...

Repository:
...

Proposed Blue defense:
...

Task:
Attempt to break this defense.
```

Therefore:

``` text
D1 → Red challenge
D3 → Red challenge
D2 → Red challenge
```

------------------------------------------------------------------------

# 17. Interaction Matrix {#17-interaction-matrix}

The final locked matrix is:

  Red Claim           Ground Truth Finding                                   Red Interaction   Blue Interaction
  ------------------- ---------------------------------------------------- ----------------- ------------------
  Attack X exists     X is valid and remains exploitable                                  +1                 -1
  Attack X exists     X is fabricated and no real attack remains                          -1                 +1
  Attack X exists     X is fabricated, but another real attack Y remains                   0                 -1
  No attack remains   Defense genuinely closes vulnerability                              +1                 +1
  No attack remains   A real vulnerability remains                                        -1                 -1

The matrix separates:

``` text
Red correctness
```

from:

``` text
Blue defense correctness
```

A broken Blue defense must not receive a positive interaction signal
merely because Red failed to identify the exact remaining vulnerability.

------------------------------------------------------------------------

# 18. D1 Example {#18-d1-example}

``` text
D1:
Verify resource ownership.
```

Red:

``` text
No attack remains.
```

Oracle:

``` text
Defense genuinely closes IDOR.
```

Outcome:

``` text
Red interaction = +1
Blue interaction = +1
```

------------------------------------------------------------------------

# 19. D3 Example {#19-d3-example}

``` text
D3:
Rotate JWT secret.
```

Red:

``` text
IDOR remains.
```

Oracle:

``` text
IDOR is valid and remains exploitable.
```

Outcome:

``` text
Red interaction = +1
Blue interaction = -1
```

Red successfully demonstrated that the defense failed.

------------------------------------------------------------------------

# 20. D2 Example {#20-d2-example}

``` text
D2:
Add rate limiting.
```

Red:

``` text
SQL Injection remains.
```

Oracle:

``` text
SQL Injection is fabricated.
IDOR remains exploitable.
```

Outcome:

``` text
Red interaction = 0
Blue interaction = -1
```

This means:

``` text
Red is not penalized for naming the wrong attack, since a real
vulnerability does remain — it just isn't the one Red identified.

Blue failed because the defense did not close the real vulnerability.
```

Neither model is rescued by the other\'s mistake.

------------------------------------------------------------------------

# 21. Episode Interaction Table {#21-episode-interaction-table}

For the example:

  Blue Defense   Red Claim               Actual State     Red Interaction   Blue Interaction
  -------------- ----------------------- -------------- ----------------- ------------------
  D1             No attack remains       IDOR closed                   +1                 +1
  D3             IDOR remains            IDOR remains                  +1                 -1
  D2             SQL Injection remains   IDOR remains                   0                 -1

The interaction rewards are attached to:

``` text
D1
D3
D2
```

not:

``` text
A1
A4
A2
```

------------------------------------------------------------------------

# 22. Interaction Advantage {#22-interaction-advantage}

For the tested Blue defenses:

``` text
D1
D3
D2
```

normalize the interaction rewards as their own group:

``` text
A_interaction(i) =
    (I_i - mean(I))
    /
    (std(I) + epsilon)
```

For untested candidates:

``` text
D4–D8:
A_interaction = 0
```

Zero means:

``` text
No interaction evidence.
```

It does not mean the defense is bad.

------------------------------------------------------------------------

# 23. Blue Advantage Fusion {#23-blue-advantage-fusion}

Blue receives two signals.

### Oracle advantage

``` text
A_oracle
```

asks:

``` text
Is this objectively a good defense?
```

### Interaction advantage {#interaction-advantage}

``` text
A_interaction
```

asks:

``` text
Did this defense survive adversarial testing?
```

Final:

``` text
A_final =
    alpha * A_oracle
    +
    beta * A_interaction
```

The interaction reward is normalized before fusion.

------------------------------------------------------------------------

# 24. Red GRPO Update {#24-red-grpo-update}

Red\'s Turn-1 group is:

``` text
A1–A8
```

Red uses:

``` text
A_oracle_red
```

for the V1 GRPO update.

There is no artificial mapping such as:

``` text
A1 → D1 interaction
A4 → D3 interaction
A2 → D2 interaction
```

because those Red candidates were not actually challenged.

Therefore:

``` text
Red Turn 1
    ↓
Oracle
    ↓
Oracle advantages
    ↓
GRPO
    ↓
Red LoRA
```

------------------------------------------------------------------------

# 25. Blue GRPO Update {#25-blue-grpo-update}

Blue\'s group is:

``` text
D1–D8
```

Each candidate receives:

``` text
A_final =
    alpha*A_oracle
    +
    beta*A_interaction
```

Therefore:

``` text
Blue Turn 2
    ↓
Oracle
    +
Turn-3 interaction
    ↓
Advantage fusion
    ↓
GRPO
    ↓
Blue LoRA
```

------------------------------------------------------------------------

# 26. Turn-3 Red Training Scope {#26-turn-3-red-training-scope}

Turn-3 Red rewards are retained:

``` text
D1 challenge → Red +1
D3 challenge → Red +1
D2 challenge → Red -1
```

They are used for:

``` text
evaluation
metrics
replay analysis
```

They do not create a second Red GRPO update in V1.

Reason:

``` text
Turn 1 = attack discovery
Turn 3 = defense verification
```

These are different generation objectives.

V1 therefore keeps:

``` text
One Red GRPO update per episode.
One Blue GRPO update per episode.
```

A dedicated Turn-3 Red objective is a future extension.

------------------------------------------------------------------------

# 27. Complete Episode {#27-complete-episode}

``` text
INPUT:
    Scenario
    Red LoRA
    Blue LoRA

1. Load scenario.

2. Extract safe scenario.md sections.

3. Load metadata internally.

4. Resolve:
       all relevant files
       2–5 noise files

5. Load selected source files.

6. Build Red Turn-1 prompt.

7. Generate G=8 Red attacks.

8. Oracle-score all Red attacks.

9. Calculate Red Oracle advantages.

10. Select Attack_best.

11. Apply confidence gate.

12. Build Blue Turn-2 prompt.

13. Generate G=8 Blue defenses.

14. Oracle-score all Blue defenses.

15. Calculate Blue Oracle advantages.

16. Select Blue Top-K.

17. For each Top-K defense:
       build Red Turn-3 challenge prompt
       generate Red challenge
       Oracle-verify the challenge
       calculate interaction outcome

18. Normalize interaction rewards.

19. Set interaction advantage to zero
    for untested Blue candidates.

20. Fuse Blue advantages.

21. Perform Red GRPO update.

22. Perform Blue GRPO update.

23. Log episode.

24. Store replay record.

25. Update weakness statistics.

26. Save checkpoint if required.

27. Sample next scenario.

28. Repeat.
```

------------------------------------------------------------------------

# 28. Complete Training Loop {#28-complete-training-loop}

``` text
Training Scenarios
        ↓
Weakness Sampler
        ↓
Scenario
        ↓
Scenario Context
        ↓
Prompt Builder
        |
        +--------------------------+
        |                          |
        v                          v
     RED TURN 1                BLUE TURN 2
     G=8 attacks               G=8 defenses
        |                          |
        v                          v
     Red Oracle                Blue Oracle
        |                          |
        v                          v
 Red Oracle Adv.             Blue Oracle Adv.
        |                          |
        v                          v
 Attack_best                  Blue Top-K
        |                          |
        |                          v
        |                     RED TURN 3
        |                          |
        |                          v
        |                       Oracle
        |                          |
        |                          v
        |                  Interaction Rewards
        |                          |
        |                          v
        |                  Interaction Adv.
        |                          |
        |                          v
        |                    Advantage Fusion
        |                          |
        v                          v
     Red GRPO                   Blue GRPO
        |                          |
        v                          v
     Red LoRA                   Blue LoRA
        |                          |
        +-------------+------------+
                      |
                      v
                Replay Buffer
                      |
                      v
               Metrics Logger
                      |
                      v
               Weakness Mining
                      |
                      v
                Next Episode
```

------------------------------------------------------------------------

# 29. Weakness Mining {#29-weakness-mining}

Replay statistics are maintained by attack/security category.

Example:

``` text
IDOR  → 0.84
JWT   → 0.72
SSRF  → 0.41
XXE   → 0.36
```

After enough observations:

``` text
Weak:
    SSRF
    XXE
```

Sampling can use:

``` text
70% → weak categories
30% → random categories
```

A minimum number of observations prevents unstable early statistics from
dominating sampling.

------------------------------------------------------------------------

# 30. Validation {#30-validation}

Validation never updates model parameters.

Validation performs the same reasoning pipeline but disables:

``` text
optimizer.step()
```

and other training updates.

Metrics are collected for:

``` text
Red attack quality
Blue defense quality
Turn-3 verification
combined validation reward
category-level performance
```

Validation data does not enter weakness mining or training updates.

------------------------------------------------------------------------

# 31. Testing {#31-testing}

The test set is held out completely.

Testing occurs after training.

``` text
Test Scenario
      ↓
Red attack generation
      ↓
Red evaluation

Test Scenario
      ↓
Blue defense generation
      ↓
Blue evaluation

Selected defenses
      ↓
Red adversarial challenge
      ↓
Interaction evaluation
```

No test reward is used for:

``` text
training
checkpoint selection
weakness sampling
```

The test set measures generalization.

------------------------------------------------------------------------

# 32. Test Metrics {#32-test-metrics}

### Red

``` text
Attack correctness
Root-cause correctness
Relevant-file F1
Oracle reward
Hallucination rate
```

### Blue

``` text
Defense correctness
Root-cause coverage
Oracle reward
Adversarial survival
```

### Turn 3

``` text
Verification accuracy
Correct bypass rate
Hallucination rate
Missed-vulnerability rate
Wrong-attack rate
```

### System

``` text
Combined validation reward
Blue adversarial robustness
Category-level performance
```

------------------------------------------------------------------------

# 33. Oracle Anti-Reward-Hacking Design {#33-oracle-anti-reward-hacking-design}

The Oracle is hybrid.

Deterministic wherever possible:

``` text
Attack label
    ↓
exact/category match

Relevant files
    ↓
set/F1 comparison

Output format
    ↓
schema/regex validation
```

Semantic judging only where necessary:

``` text
Reasoning quality
Root-cause quality
Unsupported claims
Hallucination
```

This reduces the surface for reward hacking.

------------------------------------------------------------------------

# 34. Oracle Self-Consistency {#34-oracle-self-consistency}

Semantic Oracle judgments can be repeated:

``` text
ORACLE_JUDGMENTS = 2
```

and averaged.

This reduces variance from one stochastic judge call.

------------------------------------------------------------------------

# 35. Why No Outcome Matrix {#35-why-no-outcome-matrix}

Traditional self-play might require:

``` text
A1 × D1
A1 × D2
A1 × D3
...
A8 × D8
```

SOGARL does not precompute this.

Instead:

``` text
Red attack
    ↓
Blue defense
    ↓
Red challenges actual defense
    ↓
Oracle evaluates actual interaction
```

The outcome is generated from the actual reasoning trajectory.

------------------------------------------------------------------------

# 36. Why Repository Execution Is Not Required {#36-why-repository-execution-is-not-required}

The repository is treated as a static reasoning environment.

Evidence comes from:

``` text
source code
+
safe scenario context
+
hidden metadata available only to Oracle
+
adversarial verification
```

No runnable application is required for every scenario.

------------------------------------------------------------------------

# 37. Model Architecture {#37-model-architecture}

The base model is shared conceptually, while Red and Blue have separate
trainable LoRA adapters.

``` text
Base Model
    |
    +---- Red LoRA
    |
    +---- Blue LoRA
```

The exact base model must be the same model used to create the original
SFT adapters.

------------------------------------------------------------------------

# 38. Checkpointing {#38-checkpointing}

Checkpoint contents:

``` text
checkpoint/
├── red_adapter/
├── blue_adapter/
├── optimizer_red.pt
├── optimizer_blue.pt
├── trainer_state.pt
├── rng_state.pt
└── checkpoint_info.json
```

The base model is not duplicated in every checkpoint.

This is important for Kaggle/cloud storage.

Resume flow:

``` text
Latest checkpoint
        ↓
Load Red adapter
        ↓
Load Blue adapter
        ↓
Load optimizer states
        ↓
Load trainer state
        ↓
Restore RNG state
        ↓
Continue training
```

------------------------------------------------------------------------

# 39. Kaggle / Cloud Execution {#39-kaggle--cloud-execution}

The final implementation must separate immutable inputs from writable
outputs.

Conceptually:

``` text
/kaggle/input/
    SFT dataset
    Red LoRA
    Blue LoRA
    Base model if provided

        ↓

/kaggle/working/
    SOGARL/
        outputs/
            checkpoints/
            replay/
            metrics/
            logs/
```

Paths are controlled through `config.py`.

Because Kaggle has a session/time limit, checkpoints are essential.

A long run must never depend on completing the entire experiment in one
session.

------------------------------------------------------------------------

# 40. Implementation Order {#40-implementation-order}

Implement and test in this order:

``` text
1. config.py

2. data/
   models.py
   scenario_loader.py
   metadata_loader.py
   path_resolver.py
   code_loader.py

3. utils/
   file_utils.py
   random_utils.py
   context_estimator.py
   logger.py
   checkpoint.py
   seed.py
   device.py

4. generation/
   prompt_builder.py
   generator.py

5. oracle/
   deterministic_checks.py
   interaction_checker.py
   oracle.py

6. sampling/
   weakness_sampler.py

7. logging/
   replay_buffer.py
   metrics_logger.py

8. rl/
   reward_manager.py
   grpo.py
   episode_manager.py

9. scripts/test_oracle.py

10. scripts/test_episode.py

11. scripts/validate_dataset.py

12. scripts/evaluate.py

13. scripts/train.py
```

------------------------------------------------------------------------

# 41. Pre-Training Tests {#41-pre-training-tests}

### Dataset test

Run:

``` text
validate_dataset.py
```

Verify:

``` text
scenario structure
required metadata
safe scenario sections
relevant file paths
noise selection
missing files
```

### Oracle test

Run:

``` text
test_oracle.py
```

Test:

``` text
correct attack
wrong attack
wrong attack + another real vulnerability
correct defense
broken defense
```

### Episode test

Run:

``` text
test_episode.py
```

Verify one complete:

``` text
Red
→ Blue
→ Red challenge
→ Oracle
→ reward fusion
→ GRPO
```

### Small training run

Run:

``` text
1–3 scenarios
1 epoch
small rollout count
```

Only after this succeeds should the full run begin.

------------------------------------------------------------------------

# 42. Final Reward Architecture {#42-final-reward-architecture}

``` text
RED
──────────────────────────────

G=8 attacks
      ↓
Oracle
      ↓
A_oracle_red
      ↓
Red GRPO
      ↓
Red LoRA
```

``` text
BLUE
──────────────────────────────

G=8 defenses
      ↓
Oracle
      ↓
A_oracle_blue
      ↓
Top-K
      ↓
Red adversarial challenge
      ↓
Oracle
      ↓
Interaction reward
      ↓
A_interaction
      ↓
alpha*A_oracle
+
beta*A_interaction
      ↓
Blue GRPO
      ↓
Blue LoRA
```

------------------------------------------------------------------------

# 43. Final Locked Decisions {#43-final-locked-decisions}

  Component                             V1 Decision
  ------------------------------------- ---------------------------
  Repository execution                  No
  Outcome matrix                        No
  Red Turn-1 rollouts                   G=8
  Blue Turn-2 rollouts                  G=8
  Blue adversarial Top-K                K=3
  Turn-3                                Red adversarial challenge
  Turn-3 Oracle                         Yes
  Blue interaction reward               Yes
  Interaction normalization             Yes
  Untested Blue interaction advantage   0
  Red Turn-3 GRPO                       No
  Red Turn-3 metrics                    Yes
  Confidence gate                       Category-relative
  Oracle                                Deterministic + semantic
  Oracle self-consistency               Configurable
  Weakness sampling                     70% weak / 30% random
  Validation updates                    Disabled
  Test updates                          Disabled
  Checkpointing                         Required
  Base model copied per checkpoint      No
  LoRA adapters checkpointed            Yes
  Optimizer state checkpointed          Yes
  RNG state checkpointed                Yes
  Kaggle support                        Required
  Cloud support                         Required

------------------------------------------------------------------------

# 44. Final End-to-End Flow {#44-final-end-to-end-flow}

``` text
                    TRAINING SCENARIO
                           |
                           v
                  Scenario Loader
                           |
                           v
                  Safe Scenario Context
                           |
                           v
                   Metadata / Paths
                           |
                           v
                  Relevant + Noise Code
                           |
                           v
                    Prompt Builder
                           |
                           v
                 +-------------------+
                 |    RED TURN 1     |
                 |    G = 8          |
                 +---------+---------+
                           |
                           v
                      Red Oracle
                           |
                           v
                  Red Oracle Advantage
                           |
                           v
                     Attack_best
                           |
                           v
                   Confidence Gate
                           |
                           v
                 +-------------------+
                 |   BLUE TURN 2     |
                 |    G = 8          |
                 +---------+---------+
                           |
                           v
                     Blue Oracle
                           |
                           v
                  Blue Oracle Advantage
                           |
                           v
                      Top-K = 3
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
            D1            D3            D2
             |             |             |
             v             v             v
        Red Challenge Red Challenge Red Challenge
             |             |             |
             +-------------+-------------+
                           |
                           v
                         Oracle
                           |
                           v
                  Interaction Outcomes
                           |
                           v
                 Interaction Advantages
                           |
                           v
                Blue Advantage Fusion
                           |
                           v
                    Blue Final Adv.
                           |
              +------------+------------+
              |                         |
              v                         v
          Red GRPO                  Blue GRPO
              |                         |
              v                         v
          Red LoRA                  Blue LoRA
              |                         |
              +------------+------------+
                           |
                           v
                    Replay Buffer
                           |
                           v
                  Metrics + Evaluation
                           |
                           v
                    Weakness Mining
                           |
                           v
                     Next Scenario
```

------------------------------------------------------------------------

# 45. Final Concept {#45-final-concept}

The final SOGARL system is:

``` text
Red discovers.
Blue defends.
Red attacks the defense.
Oracle grounds the interaction.
GRPO improves both policies.
Weakness mining creates a harder curriculum.
```

The critical distinction is:

``` text
Red Turn 1
    → trains Red

Blue Turn 2
    → trains Blue

Red Turn 3
    → adversarially tests Blue
    → determines Blue interaction reward
    → evaluates Red
    → does not create a second Red GRPO update in V1
```

This gives SOGARL a genuine adversarial interaction loop without
requiring:

``` text
repository execution
```

or:

``` text
a manually constructed attack × defense outcome matrix.
```