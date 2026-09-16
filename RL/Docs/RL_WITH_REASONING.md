# Independent RL With Reasoning

**Consolidated documentation: 16 September 2026**

Read [Complete RL process](RL_COMPLETE_PROCESS.md), [RL without reasoning](RL_WITHOUT_REASONING.md), or [RL with reasoning](RL_WITH_REASONING.md).

> Configuration values describe notebook defaults, not every completed experiment. Use the saved run configuration when interpreting results. In particular, the standard notebook defaults to 6,144 input tokens; a specific non-reasoning experiment may override this to 8,192. Successful optimizer updates must be counted from logs because skipped groups do not update the policy.

This document describes the current context-and-reasoning
reinforcement-learning approach implemented in:

-   `red_team_context_reasoning_grpo.ipynb`
-   `blue_team_context_reasoning_grpo.ipynb`

It supersedes earlier descriptions that treated reasoning as a future
idea. In the current notebooks, reasoning is enabled through
`REASONING_ENABLED=True`, and the Red and Blue policies are trained
independently.

## 1. Executive summary

The project starts with two separately supervised-fine-tuned LoRA
adapters:

-   **Red Team:** identifies security attacks supported by the supplied
    repository code.
-   **Blue Team:** identifies defenses supported by the supplied
    repository code.

The new RL approach keeps the existing supervised adapters and applies a
custom **GRPO-style policy-optimization loop** on top of them.

The important change is that the model is now allowed to produce a
short, code-grounded explanation before its structured answer.

The generated response has two conceptual parts:

``` text
REASONING:
Brief explanation grounded in supplied code.

FINAL:
valid_attacks: ["action_name"]
relevant_files: ["files/example.py"]
```

For Blue, `valid_attacks` becomes `valid_defenses`.

The evaluator **scores only the FINAL section**. The reasoning section
is not independently graded against a reasoning reference. Nevertheless,
because the entire completion participates in the policy loss, a
high-reward response provides an indirect learning signal to both
reasoning and final-answer tokens.

This is therefore **outcome-based reasoning training**, not supervised
chain-of-thought training and not a reasoning-verification system.

## 2. Overall system architecture

```text
Scenario source -> Context builder -> Role policy -> Five completions
                                         ^                 |
Role SFT adapter -> Trainable policy -----+                 v
                 -> Frozen reference                 Final parser
                         |                                 |
                         v                                 v
                  Reference probabilities      Scorer <- Hidden metadata
                         |                                 |
                         +-> GRPO objective <- Group-relative advantages
                                     |
                                Role LoRA update
                                     |
                         Periodic validation and ranking
                                     |
                         Role-specific adapter exports
```

This workflow is instantiated separately for Red and Blue. Metadata enters only the scorer; it does not enter the policy prompt.


Red and Blue are separate pipelines. There is no Red → Blue → Red
interaction.

## 3. What changed from the standard approach?

The standard notebooks remain useful as a baseline. The
context-and-reasoning notebooks introduce several concrete implementation
changes.

  -----------------------------------------------------------------------
  Component               Standard                Context + reasoning
  ----------------------- ----------------------- -----------------------
  Input budget            6144 tokens             8192 tokens

  Reasoning               Disabled/not requested  Enabled

  Output budget           384 tokens              768 tokens

  Capacity check          No disposable trial     3 scenarios × 5
                                                  rollouts

  Path handling           Basic normalization     Repository-aware unique
                                                  path recovery

  Advantage normalization `std + 1e-6`            `max(std, 0.05)`

  Reward                  0.8 action + 0.2 files  Same

  Optimizer               AdamW, 5e-6             Same

  GRPO loop               Custom                  Custom

  Reference model         Frozen SFT reference    Frozen SFT reference
  -----------------------------------------------------------------------

The most important conceptual point is that **reasoning does not
introduce a new reward function**. The reward still comes from the final
attack/defense and relevant-file predictions.

## 4. Inputs and scenario preparation

The notebooks search under `/kaggle/input` for the scenario dataset and
the corresponding SFT adapter.

Overrides are available when automatic discovery is ambiguous:

``` text
DATASET_OVERRIDE
ADAPTER_OVERRIDE
```

The adapter directory must contain:

``` text
adapter_config.json
adapter_model.safetensors
```

The base model is resolved from the adapter configuration. The RL stage
therefore continues training the existing SFT LoRA adapter rather than
starting from a fresh LoRA.

Only:

``` text
scenario_001 ... scenario_100
```

are considered.

A seed of `42` produces:

-   80 training scenarios
-   10 validation scenarios
-   10 test scenarios

The split is created before later data-quality checks. Missing or
unreadable referenced files can still remove a record and are reported
rather than silently ignored.

## 5. What the model sees vs. what the evaluator sees

This separation is fundamental.

## Model input

The model receives:

-   repository/file inventory
-   available source code
-   repository paths
-   task instructions
-   role-specific output requirements

The scenario description is omitted because it may reveal the intended
answer.

## Evaluator input

The evaluator separately retains:

-   expected attacks or defenses
-   expected relevant files

The metadata answer is **not inserted into the model prompt**.

Therefore:

``` text
Code/context -> Model -> Prediction
Metadata      -> Evaluator -> Reward
```

This prevents direct answer leakage.

## 6. Context construction

The new approach increases the input budget to **8192 tokens**.

If the entire repository fits, the complete source is retained.

If it does not fit, the context builder:

1.  keeps the repository file inventory;
2.  allocates a source-token budget across files;
3.  redistributes unused budget from shorter files;
4.  retains the beginning and end of oversized files;
5.  inserts an omission marker for removed middle content;
6.  re-tokenizes the complete rendered prompt;
7.  searches for an allocation that fits the total input budget.

The selection is not target-aware.

This is important: the system does **not** inspect the hidden metadata
and then select the files containing the answer.

### Current limitation

The strategy is balanced **head/tail truncation**, not function-aware
retrieval.

Therefore a decisive function located in the middle of a long file can
still be omitted.

## 7. Reasoning-enabled generation

With:

``` text
REASONING_ENABLED = True
```

the model is instructed to first give a short explanation grounded in
the supplied source code.

The intended format is:

``` text
REASONING:
<brief mechanism explanation>

FINAL:
valid_attacks: [...]
relevant_files: [...]
```

Blue uses:

``` text
FINAL:
valid_defenses: [...]
relevant_files: [...]
```

The reasoning budget is **768 output tokens for the complete response**,
not 768 tokens for reasoning plus another 768 for the answer.

The prompt asks for approximately at most 150 words of reasoning. That
is an instruction, not a hard token-level enforcement mechanism.

## 8. Why the FINAL boundary matters

The evaluator must not accidentally treat reasoning text as an answer.

The parser therefore searches for a unique standalone final heading.

Accepted forms include:

-   `FINAL`
-   `FINAL:`
-   `FINAL ANSWER`
-   common Markdown heading/bold forms
-   case-insensitive equivalents

Only text after that boundary is passed to the component scorer.

Consequences:

### Valid

``` text
REASONING:
The request flow reaches the audit logger...

FINAL:
valid_attacks: ["audit_log_injection"]
relevant_files: ["src/audit_logger.rs"]
```

### Invalid

If the model mentions a second `FINAL` section or never provides a
recognizable final heading, extraction fails.

### Important asymmetry

Missing reasoning does **not** invalidate a structurally valid final
answer.

A model can therefore learn to produce a useful final answer even if it
does not generate a reasoning section.

## 9. How reasoning actually learns

There is no reasoning answer key.

The reward does not ask:

> "Is this explanation logically correct?"

It asks whether the final predicted actions and files match the
evaluator reference.

Suppose five completions receive:

``` text
0.20
0.80
0.10
0.70
0.00
```

The better responses receive positive relative advantages and the worse
responses receive negative relative advantages.

Because the generated completion contains both:

``` text
REASONING + FINAL
```

the scalar advantage is applied across completion tokens.

Therefore a high-reward response indirectly reinforces the reasoning
tokens that occurred in that response.

This is **credit through outcome**, not direct reasoning supervision.

### Consequence

A fluent but incorrect explanation can receive learning pressure if its
final answer is correct.

Conversely, a logically good explanation attached to an incorrect final
answer can receive a poor reward.

The current implementation does not verify the causal correctness of the
explanation.

## 10. Path recovery improvement

The new scorer receives the actual repository inventory.

It first prefers exact path matches.

If necessary, it attempts unique recovery for:

-   a leading `/`
-   omitted `files/`
-   omitted `codebase/`
-   unambiguous capitalization differences

For example, if the repository contains:

``` text
files/src/audit_logger.rs
```

and the model predicts an unambiguous equivalent such as:

``` text
src/audit_logger.rs
```

the scorer can resolve it.

Ambiguous matches are not guessed.

Resolved aliases are deduplicated and logged with:

-   original path
-   resolved path
-   resolution status

Whitespace-separated unquoted paths can also be recovered when every
separated token clearly resembles a repository path with an extension.

This makes file scoring less sensitive to formatting/path-prefix
mistakes while still rejecting arbitrary substring matching.

## 11. Reward function

The reward remains:

``` text
reward = 0.8 * action_F1 + 0.2 * file_F1
```

## Action matching

Action names are normalized by:

1.  lowercasing;
2.  replacing `_` and `-` with spaces;
3.  collapsing whitespace.

Exact normalized matches receive full credit.

Remaining predictions and targets are compared using embeddings from:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are normalized and cached.

For a predicted/target pair:

``` text
pair_credit =
    clip((cosine_similarity - 0.60) / (1 - 0.60), 0, 1)
```

Hungarian assignment chooses the one-to-one matching with maximum total
credit.

This prevents several predictions from all claiming the same target.

## Action F1

``` text
action_F1 =
    2 * sum(assigned_pair_credits)
    / (number_predicted + number_target)
```

## File F1

``` text
file_F1 =
    2 * number_matching_paths
    / (number_predicted_paths + number_target_paths)
```

The file component is therefore worth 20% of the final reward.

## 12. Extraction failures are different from empty answers

An empty list can be a legitimate prediction.

An unextractable field is not treated as an empty list.

Therefore:

``` text
missing/unextractable action field -> action contribution = 0
missing/unextractable file field   -> file contribution = 0
```

If generation does not end with an accepted EOS condition, the final
total reward is gated to zero even if some fields were successfully
extracted.

This prevents incomplete generations from receiving partial reward
merely because a useful fragment appeared before generation stopped.

## 13. Group-relative advantage

For each scenario, five responses form a group.

The new implementation uses:

``` text
advantage_i =
    (reward_i - group_mean)
    / max(population_std, 0.05)
```

The `0.05` value is an **advantage floor**.

### Why?

Suppose rewards are almost identical:

``` text
0.500
0.501
0.500
0.499
0.500
```

Without a floor, the tiny differences can be divided by a very small
standard deviation and create disproportionately large advantages.

The floor reduces that amplification.

It does not:

-   improve raw reward;
-   create correctness;
-   remove file-driven learning;
-   guarantee better convergence.

If reward variance is below `1e-6`, the group update is skipped.

## 14. The GRPO-style optimization loop

This is a custom implementation rather than TRL's `GRPOTrainer`.

For each training scenario:

``` text
1. Build code context
2. Generate 5 independent responses
3. Extract FINAL sections
4. Recover/normalize paths
5. Compare actions and files with metadata
6. Calculate 5 rewards
7. Calculate group advantages
8. Compute old-policy log probabilities
9. Compute frozen-reference log probabilities
10. Compute current-policy log probabilities
11. Calculate clipped policy objective + KL
12. Accumulate gradients across the 5 responses
13. Clip gradients
14. Perform one optimizer update
```

One scenario therefore produces **one optimizer update**, not five.

With 80 training scenarios and one epoch:

``` text
80 episodes
5 responses/episode
= 400 sampled training responses
```

Validation generations do not update the policy.

## 15. Policy/reference relationship

Two adapter states are involved:

### Trainable policy

The current SFT-derived LoRA adapter.

### Frozen reference

A copy of the original SFT weights.

The reference is used to discourage uncontrolled policy drift.

For each response:

``` text
ratio = exp(current_logp - old_logp)

delta = reference_logp - current_logp

sampled_KL = exp(delta) - delta - 1
```

The clipped policy objective is:

``` text
min(
    ratio * advantage,
    clip(ratio, 0.8, 1.2) * advantage
)
```

The final token-level loss is:

``` text
loss =
    sum(
        -objective
        + 0.02 * sampled_KL
    )
    / total_completion_tokens
```

The scalar group advantage is repeated across completion tokens.

### Important consequence

Longer completions contain more completion tokens and therefore
contribute more terms to the accumulated group loss.

This is not equivalent to giving every response an equal-weight mean
loss.

## 16. Sampling and optimizer configuration

The current notebooks use:

``` text
learning rate = 5e-6
AdamW
temperature = 1
top_p = 1
top_k = 0
clip epsilon = 0.2
KL coefficient = 0.02
```

The sampling configuration is intentionally matched to the unwarped
policy used for log-probability scoring.

The base model is loaded through Unsloth using:

-   4-bit base weights
-   gradient checkpointing
-   one CUDA device
-   FP32 trainable policy parameters
-   BF16 when supported
-   otherwise FP16 + AMP scaling
-   dropout disabled for stable probability comparisons

Two T4 GPUs are not automatically combined into one shared memory pool.

## 17. Numerical safety

FP16 AMP starts with a scale of 256.

If gradients are nonfinite:

1.  the unsafe optimizer step is skipped;
2.  AMP lowers the scale;
3.  the same rollouts are retried;
4.  up to two retries are allowed.

Finite gradient norms are accumulated in float64 and clipped to 1.

Repeated exhausted overflow groups stop training.

A nonfinite loss raises an error rather than being silently accepted.

## 18. Disposable capacity trial

The new context notebooks have:

``` text
TRIAL_MODE = True
```

by default.

The trial is intentionally disposable.

It selects three training scenarios with the largest original prompt
lengths and generates five responses for each.

So the capacity workload is:

``` text
3 scenarios × 5 rollouts
```

The trial executes the real expensive path:

``` text
generation
    -> reward
    -> log probabilities
    -> backward
    -> optimizer update
```

This is important because successful generation alone does not prove
that the backward pass will fit in GPU memory.

Before the trial:

-   trainable policy weights are copied to CPU;
-   optimizer state is saved;
-   scaler state is saved.

Afterward, a `finally` block restores them.

No trained adapter ZIP is produced by the trial.

### Zero-variance special case

If actual rewards have essentially zero variance, the trial may
temporarily use:

``` text
0, 1, 2, 3, 4
```

as synthetic advantages solely to force the backward-memory path.

The event is explicitly logged as:

``` text
synthetic_memory_test_advantages = true
```

The actual reported reward remains based on real outputs.

The synthetic update is discarded.

## 19. Correct way to run the new notebooks

## Phase A --- capacity trial

Keep:

``` text
TRIAL_MODE = True
REASONING_ENABLED = True
input = 8192
output = 768
```

Run all cells in order.

Inspect:

``` text
context_trial.jsonl
scenario reports
memory events
coverage reports
```

A successful trial requires a finite optimizer update for every trial
group.

## Phase B --- clean restart

After a successful trial:

1.  restart the Kaggle session;
2.  reload the notebook;
3.  set:

``` text
TRIAL_MODE = False
REASONING_ENABLED = True
```

4.  retain the 8192/768 budgets;
5.  run the full training process.

Restarting matters because the trial intentionally modifies
model/optimizer state before restoring it, and a clean session avoids
stale runtime configuration.

## 20. Validation and checkpoint selection

Ten validation scenarios are used.

Before full training, greedy validation is performed.

If the baseline cannot produce an extractable action/defense or file
component, training stops.

Validation repeats every five training episodes and at epoch end when
needed.

The validation reward determines which policy states are retained.

Only:

``` text
best_adapter.zip
second_best_adapter.zip
```

are retained.

A temporary export contains:

-   policy LoRA weights
-   tokenizer
-   ranking metadata

and is removed after ZIP creation.

The ZIP is **not**:

-   a merged standalone model;
-   a full optimizer-resume checkpoint;
-   a complete copy of the base model.

The base model remains required for inference.

## 21. What the saved models mean

The saved adapter is a role-specific LoRA update optimized for:

``` text
source code -> supported attacks/defenses + relevant files
```

It is not a universal security verifier.

A high validation reward means:

> the generated final fields matched the deterministic evaluator well
> under the current scoring function.

It does **not** prove:

-   that the vulnerability is executable;
-   that the attack is actually exploitable;
-   that the defense is sufficient;
-   that the reasoning is causally correct;
-   that the model generalizes to repositories never represented during
    SFT.

## 22. Logging and experiment inspection

Each run creates a timestamped directory.

Important artifacts include:

-   configuration
-   dependency versions
-   console output
-   scenario discovery
-   context coverage
-   prompts
-   sampled rollouts
-   rewards
-   advantages
-   token-level optimization details
-   validation metrics
-   scenario summaries
-   checkpoint/save events

Per-scenario reports contain the input/reference separation, context
coverage, responses, and optimization information.

`index.html` provides a report index.

For the reasoning approach, inspect the reasoning and final-answer
boundary explicitly. A model can produce plausible reasoning while
failing the structured final parser.

## 23. Red and Blue are independent

The architecture is:

``` text
Repository
   |
   +----> Red SFT adapter -> Red GRPO -> Red adapter
   |
   +----> Blue SFT adapter -> Blue GRPO -> Blue adapter
```

Each role owns its adapter, reference, optimizer and logs. The two training loops operate independently.

The metadata evaluator remains the source of reward for both roles.

### Red

``` text
valid_attacks
relevant_files
```

### Blue

``` text
valid_defenses
relevant_files
```

## 24. Implementation differences

The current approach improves the implementation in several concrete
ways:

1.  **More source context:** 8192-token input budget instead of 6144.
2.  **Reasoning-enabled output:** the model can articulate the mechanism
    before the final answer.
3.  **Strict final boundary:** reasoning is prevented from being
    accidentally scored as prediction.
4.  **Repository-aware path recovery:** common
    path-prefix/capitalization differences are handled when unambiguous.
5.  **Advantage floor:** tiny reward differences are less likely to
    create exaggerated updates.
6.  **Disposable capacity test:** expensive generation + backward +
    optimizer behavior is checked before full training.
7.  **Separate logging:** coverage, reasoning, final extraction, rewards
    and optimization behavior can be audited.

These are engineering improvements. They are not evidence by themselves
that the model is a better security analyst.

## 25. Remaining limitations

The current system still has important weaknesses.

## 25.1 Context truncation

Head/tail selection can omit decisive middle-of-file functions.

## 25.2 Semantic action matching

MiniLM similarity is a heuristic. The `0.60` threshold is provisional
and not calibrated proof of equivalence.

## 25.3 No executable verification

The system does not run code, exploit candidates, or verify defenses.

## 25.4 No reasoning judge

Reasoning is not independently checked.

## 25.5 No separate action/file advantages

A single combined reward advantage controls the whole completion.

## 25.6 Token-length weighting

Longer completions contribute more token-level loss terms.

## 25.7 Validation size

Ten validation scenarios provide limited evidence about generalization.

## 25.8 SFT leakage uncertainty

A scenario held out from GRPO may already have appeared during SFT. The
notebooks do not prove otherwise.

## 25.9 One-GPU constraint

The implementation does not automatically pool two T4 cards into one
memory space.

## 26. The complete learning loop in one picture

``` text
                 START
                   |
                   v
          Existing SFT adapter
                   |
                   v
        Select training scenario
                   |
                   v
          Build repository context
                   |
                   v
          Generate 5 completions
                   |
                   v
       +-------------------------+
       | REASONING + FINAL       |
       +-------------------------+
                   |
                   v
        Extract FINAL only
                   |
          +--------+--------+
          |                 |
       actions             files
          |                 |
          +--------+--------+
                   |
                   v
          Compare with metadata
                   |
                   v
              Reward
        0.8 action + 0.2 file
                   |
                   v
        Compute group advantages
                   |
                   v
      Current / old / reference logp
                   |
                   v
        PPO/GRPO-style clipped loss
              + KL penalty
                   |
                   v
          Backward + clip grads
                   |
                   v
             AdamW update
                   |
                   v
          Next training scenario
                   |
                   v
              Validation
                   |
                   v
       Rank policy checkpoints
                   |
          +--------+--------+
          |                 |
        Best             Second best
          |                 |
          v                 v
       LoRA ZIP           LoRA ZIP
```

## 27. Final interpretation

The new approach should be described accurately as:

> **Independent Red/Blue custom GRPO-style reinforcement learning
> starting from role-specific SFT LoRA adapters, using repository code
> as context, metadata-based deterministic outcome rewards, and
> reasoning-enabled generation whose complete completion receives
> outcome-based policy-gradient credit while only the structured FINAL
> section is evaluated.**

That description is more precise than saying simply that the model is
"trained to reason."

The model is **allowed to reason and receives indirect outcome credit
for reasoning tokens**, but there is currently no independent reasoning
correctness signal.

The strongest future improvement would therefore be to make the
reasoning itself verifiable---through code-grounded checks, evidence
extraction, execution/static analysis, or a carefully designed
independent judge---rather than merely increasing the reasoning token
budget.

## Model-agnostic notebook variants

The same reasoning workflow is also provided in:

- [Red model-agnostic notebook](../Scripts/Red_RL/red_team_context_reasoning_grpo_model_agnostic.ipynb)
- [Blue model-agnostic notebook](../Scripts/Blue_RL/blue_team_context_reasoning_grpo_model_agnostic.ipynb)

These are implementation variants within this approach, not a separate learning phase. Resolve the base model from the supplied adapter and preserve compatible tokenizer/chat formatting. Record the actual notebook and configuration for every comparison. The local variants default to `TRIAL_MODE=True`, `REASONING_ENABLED=True`, and an 8,192-token prompt budget.

## Test-policy and reasoning-presence checks

A report named `test_last_policy.json` evaluates the final in-memory policy. It does not establish the test performance of the saved best adapter. Reload and evaluate the selected adapter when reporting its test results.

A successful structured parse does not prove that an explanation was produced. Inspect raw completions and explicit reasoning boundaries; a populated logging field alone is insufficient. The scorer may accept final fields without reasoning. Consequently, report explanation presence separately from format pass and reward.
