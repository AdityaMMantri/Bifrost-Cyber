# Independent RL Without Reasoning

**Consolidated documentation: 16 September 2026**

Read [Complete RL process](RL_COMPLETE_PROCESS.md), [RL without reasoning](RL_WITHOUT_REASONING.md), or [RL with reasoning](RL_WITH_REASONING.md).

> Configuration values describe notebook defaults, not every completed experiment. Use the saved run configuration when interpreting results. In particular, the standard notebook defaults to 6,144 input tokens; a specific non-reasoning experiment may override this to 8,192. Successful optimizer updates must be counted from logs because skipped groups do not update the policy.

This document describes the baseline implementation in:

-   `red_team_unsloth_grpo.ipynb`
-   `blue_team_unsloth_grpo.ipynb`

It should be treated as the **standard/baseline branch** against which
the newer context-and-reasoning approach is compared.

## 1. Purpose

The standard pipeline starts from two independently
supervised-fine-tuned LoRA adapters:

-   Red predicts supported attacks.
-   Blue predicts supported defenses.

Each policy is trained independently with a custom GRPO-style loop.

The model receives repository source code and predicts:

``` text
valid_attacks: [...]
relevant_files: [...]
```

or, for Blue:

``` text
valid_defenses: [...]
relevant_files: [...]
```

No reasoning section is requested in this baseline.

## 2. Baseline configuration

  Setting                     Standard baseline
  ------------------------- -------------------
  Input budget                      6144 tokens
  Output budget                      384 tokens
  Rollouts per scenario                       5
  Epochs                                      1
  Clip epsilon                              0.2
  KL coefficient                           0.02
  Learning rate                            5e-6
  Validation interval                5 episodes
  Advantage normalization          `std + 1e-6`

These are the defaults implemented in the attached baseline notebooks.

## 3. Independent Red and Blue learning

The two policies do not interact.

``` text
Red SFT adapter  -> Red GRPO  -> Red LoRA
Blue SFT adapter -> Blue GRPO -> Blue LoRA
```

Each role maintains its own policy, frozen reference, and optimizer state.

The deterministic scenario metadata supplies the evaluator reference.

## 4. Dataset and SFT continuation

Only `scenario_001` through `scenario_100` are selected.

Seed 42 creates:

-   80 training
-   10 validation
-   10 test

The RL stage continues the discovered role-specific SFT adapter.

The adapter directory must contain:

``` text
adapter_config.json
adapter_model.safetensors
```

The base model is resolved from the adapter configuration.

## 5. Prompt and output

The standard model is instructed to return only the structured answer.

Red:

``` text
valid_attacks: ["action_name"]
relevant_files: ["files/example.py"]
```

Blue:

``` text
valid_defenses: ["defense_name"]
relevant_files: ["files/example.py"]
```

The parser is tolerant of several structured aliases and common
formatting variants, but incomplete arrays and conflicting fields remain
extraction failures.

Unlike the context-and-reasoning version, there is no reasoning/final
boundary.

## 6. Context construction

The standard input budget is 6144 tokens.

If the repository is too large, source is reduced using balanced
head/tail truncation with coverage logging.

The context builder does not use hidden target metadata to choose files.

This means the baseline has the same fundamental context-selection
limitation as the reasoning branch: decisive middle-of-file code can be
omitted.

## 7. Reward

The baseline uses:

``` text
reward = 0.8 * action_F1 + 0.2 * file_F1
```

Actions are normalized and compared semantically using frozen:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

The semantic pair threshold is:

``` text
0.60 cosine similarity
```

Hungarian assignment creates one-to-one prediction/target matching.

File paths receive F1 credit after basic normalization.

An unextractable action or file component contributes zero
independently.

Incomplete generation that does not satisfy the accepted EOS condition
gates the final reward to zero.

## 8. Group-relative learning

Five responses are sampled for each training scenario.

Their rewards are converted into relative advantages:

``` text
advantage_i =
    (reward_i - group_mean)
    / (population_std + 1e-6)
```

There is no standard-deviation floor.

Therefore very small reward differences can produce relatively large
advantages when the group standard deviation is tiny.

If the reward variance is effectively zero, the update is skipped.

## 9. Custom GRPO-style objective

This is not TRL's `GRPOTrainer`.

For each completion:

``` text
ratio = exp(current_logp - old_logp)

delta = reference_logp - current_logp

sampled_KL = exp(delta) - delta - 1
```

The clipped objective is:

``` text
min(
    ratio * advantage,
    clip(ratio, 0.8, 1.2) * advantage
)
```

The loss is:

``` text
loss =
    sum(
        -objective
        + 0.02 * sampled_KL
    )
    / total_completion_tokens
```

The advantage is repeated across completion tokens.

Consequently, longer completions contribute more token-level loss terms.

## 10. Model and optimizer

Unsloth loads:

-   4-bit base model
-   existing LoRA adapter
-   gradient checkpointing
-   one CUDA device

A frozen reference adapter is created from the original SFT state.

The trainable policy uses FP32 parameters.

BF16 is preferred when supported; otherwise FP16 AMP is used.

Optimizer:

``` text
AdamW
learning rate = 5e-6
```

Gradient norm is clipped to 1.

Nonfinite gradients trigger AMP scaling/retry behavior.

## 11. Training episode

One episode is:

``` text
scenario
   -> 5 sampled responses
   -> extraction
   -> reward
   -> group advantages
   -> log-probability calculation
   -> backward
   -> one optimizer update
```

One epoch normally contains:

``` text
80 episodes
400 sampled training responses
```

Evaluation generations do not update the policy.

## 12. Validation

Ten fixed validation scenarios are used.

Validation runs before training and repeatedly during training.

Validation is used to rank updated policy states.

The best and second-best eligible states are retained.

Only:

``` text
best_adapter.zip
second_best_adapter.zip
```

are saved.

These are LoRA exports, not merged standalone models or exact
optimizer-resume checkpoints.

## 13. Logs

Runs produce timestamped output directories containing:

-   configuration
-   dependency versions
-   scenario discovery
-   context coverage
-   prompts
-   rollouts
-   reward/advantage data
-   optimization information
-   validation results
-   scenario summaries
-   checkpoint ranking information

Per-scenario reports and `index.html` support inspection.

## 14. Baseline limitations

The standard branch has several known limitations:

1.  6144-token input budget can remove important source context.
2.  Head/tail truncation is not function-aware.
3.  MiniLM cosine matching is only a semantic heuristic.
4.  The 0.60 threshold is provisional.
5.  No executable vulnerability verification exists.
6.  No reasoning correctness signal exists.
7.  No independent reasoning section is generated.
8.  There is no separate action/file advantage.
9.  Tiny reward variance can amplify advantages.
10. Ten validation scenarios provide limited generalization evidence.
11. Held-out GRPO scenarios may have appeared during SFT.
12. Two T4 GPUs are not automatically pooled into one memory space.

## 15. Why this baseline is still important

The standard branch is not obsolete.

It provides a controlled comparison for the new context-and-reasoning
branch.

A meaningful experiment should keep as many factors as possible constant
and compare:

``` text
same SFT initialization
same scenario split
same reward evaluator
same optimizer
same number of rollouts
same validation protocol
```

while changing the intended intervention:

``` text
standard:
6144 input + 384 output + answer only

new:
8192 input + 768 output + reasoning + FINAL boundary
                 + path recovery
                 + advantage floor
                 + disposable capacity trial
```

If the new approach improves validation reward, the result should still
be interpreted as improvement under the same deterministic
evaluator---not automatic proof of better real-world security reasoning.

## Running and interpreting this branch

1. Select the matching Red or Blue standard notebook and its existing SFT adapter.
2. Verify dataset discovery, adapter compatibility, scenario membership and actual input/output budgets.
3. Inspect the before-RL validation report and extraction/completion status.
4. Run the training groups; inspect rewards, advantages, skipped groups and successful optimizer steps.
5. Review periodic validation and the ranked adapter exports.
6. Identify the policy used for the test report. The notebooks include `test_last_policy.json`; this evaluates the last in-memory policy, not automatically the best exported adapter.

The default 80 training scenarios and five samples yield 400 training completions. This is not 400 optimizer steps. An eligible group contributes one step; zero-variance or numerical failures may skip it.

With ten validation cases, a baseline plus sixteen five-episode validations yields 170 validation completions. Adding ten test completions gives 580 total completions only when that complete schedule ran, excluding retries or additional evaluations. Validate totals against the run logs.

A positive reward can arise solely from file localization. For example, action F1 = 0 and file F1 = 0.5 give reward = 0.1. Count action-positive scenarios separately from reward-positive scenarios. Format pass indicates parseability rather than correctness.
