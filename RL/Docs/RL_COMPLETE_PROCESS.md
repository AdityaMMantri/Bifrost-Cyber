# Independent RL: Complete Process

**Consolidated documentation: 16 September 2026**

Read [Complete RL process](RL_COMPLETE_PROCESS.md), [RL without reasoning](RL_WITHOUT_REASONING.md), or [RL with reasoning](RL_WITH_REASONING.md).

> Configuration values describe notebook defaults, not every completed experiment. Use the saved run configuration when interpreting results. In particular, the standard notebook defaults to 6,144 input tokens; a specific non-reasoning experiment may override this to 8,192. Successful optimizer updates must be counted from logs because skipped groups do not update the policy.

This document explains the complete pipeline from SFT output to the
current context-and-reasoning GRPO experiments, including what happens
inside one training episode, how rewards are produced, how reasoning is
treated, how Red and Blue remain independent, how the capacity trial
works, and how to interpret the resulting adapters.

Source basis:

-   [RL without reasoning](RL_WITHOUT_REASONING.md)
-   [RL with reasoning](RL_WITH_REASONING.md)
-   `red_team_unsloth_grpo.ipynb`
-   `blue_team_unsloth_grpo.ipynb`
-   `red_team_context_reasoning_grpo.ipynb`
-   `blue_team_context_reasoning_grpo.ipynb`

## 1. Project position: SFT first, RL second

The RL stage is not the first time the model learns the task.

The intended pipeline is:

``` text
Compatible pretrained base model
       |
       v
Supervised Fine-Tuning
       |
       +--------------------+
       |                    |
       v                    v
Red SFT LoRA            Blue SFT LoRA
       |                    |
       v                    v
Red GRPO                Blue GRPO
       |                    |
       v                    v
Red RL adapter          Blue RL adapter
```

SFT teaches the model the general task format and task behavior.

RL then tries to improve the policy using relative outcome feedback.

The RL stage therefore **continues from the SFT adapters**. It does not
replace them with a newly initialized LoRA.

## 2. What Red and Blue actually learn

## Red

Red is given repository code and asked:

> Which supported attacks are evidenced by this code, and which files
> are relevant?

Its canonical output fields are:

``` text
valid_attacks
relevant_files
```

## Blue

Blue receives the analogous repository evidence and predicts:

``` text
valid_defenses
relevant_files
```

The roles are separate.

Each role maintains its own policy, reference adapter, optimizer state, and output directory. Neither role consumes the other role's predictions.

## 3. Why metadata is present but hidden from the model

Each scenario contains both:

``` text
repository/source files
metadata/reference answer
```

The model must reason from source code.

The metadata is reserved for the evaluator.

``` text
                    Scenario
                       |
             +---------+---------+
             |                   |
             v                   v
        Code context         Ground truth
             |                   |
             v                   v
          Model              Evaluator
             |                   |
             +---------+---------+
                       |
                       v
                     Reward
```

If metadata were inserted into the model prompt, the RL task would
become substantially less meaningful because the model could simply
reproduce the target.

Therefore the reference is evaluator-only.

## 4. Scenario split

The notebooks consider:

``` text
scenario_001 ... scenario_100
```

Using seed 42:

``` text
80 train
10 validation
10 test
```

The important caveat is that this is an RL split, not necessarily a true
unseen-data split.

If the SFT dataset already contained one of the validation/test
scenarios, the RL experiment does not prove repository-level
generalization to unseen scenarios.

## 5. Standard vs new experiment

The baseline asks for only the final structured answer.

``` text
valid_attacks: [...]
relevant_files: [...]
```

The new branch asks for:

``` text
REASONING:
<short code-grounded explanation>

FINAL:
valid_attacks: [...]
relevant_files: [...]
```

Blue substitutes `valid_defenses`.

The new branch also changes the computational envelope:

``` text
6144 input  -> 8192 input
384 output  -> 768 output
no trial    -> disposable capacity trial
basic paths -> repository-aware path recovery
std+1e-6    -> max(std, 0.05)
```

The underlying reward and core GRPO-style objective remain largely the
same.

## 6. New reasoning approach: what "reasoning" means here

The word "reasoning" needs to be used precisely.

The system does **not** have:

-   a gold reasoning trace;
-   a reasoning classifier;
-   a separate reasoning reward;
-   a proof checker;
-   an independent chain-of-thought judge.

Instead, it has:

``` text
code
  -> generated explanation
  -> final prediction
  -> final prediction reward
  -> policy update over the whole completion
```

Therefore the model's reasoning can receive indirect learning pressure
because the advantage is applied to the generated completion.

This is outcome-based reasoning learning.

## 7. Why the FINAL section is necessary

Consider a response:

``` text
The cache loader accepts untrusted serialized state.
This can allow object injection.

valid_attacks: ["deserialization_attack"]
relevant_files: ["src/cache_loader.java"]
```

If the parser simply searched the entire response, words inside the
explanation could accidentally be treated as predictions.

The new implementation creates an explicit boundary:

``` text
REASONING:
...
FINAL:
...
```

Only the final section is scored.

This creates a clean separation:

``` text
Reasoning = explanatory generation
Final     = prediction interface
```

This is one of the most important changes in the new approach.

## 8. What happens during inference/generation

For each scenario:

1.  Repository files are discovered.
2.  The context builder creates the model input.
3.  Five independent completions are sampled.
4.  Each completion can contain reasoning and a final answer.
5.  The parser locates the final section.
6.  Actions/defenses and files are extracted.
7.  The evaluator computes the reward.

The model never receives the hidden answer during this process.

## 9. Five rollouts are the GRPO comparison group

Why five?

Because the RL signal is relative.

Suppose:

``` text
Response A -> 0.90
Response B -> 0.70
Response C -> 0.20
Response D -> 0.10
Response E -> 0.80
```

The optimizer can learn:

``` text
A, E, B > C, D
```

for that scenario.

The group therefore provides a local ranking signal.

This is different from ordinary supervised learning, where the model is
given one target output and trained directly against it.

## 10. Reward calculation in detail

The reward has two components.

``` text
80% action correctness
20% relevant-file correctness
```

Formally:

``` text
R = 0.8 A_F1 + 0.2 F_F1
```

## Action normalization

For matching:

``` text
lowercase
underscore/hyphen -> spaces
collapse whitespace
```

Exact normalized matches receive full credit.

Remaining pairs use MiniLM cosine similarity.

The current threshold is:

``` text
0.60
```

Partial pair credit is:

``` text
clip((cosine - 0.60) / 0.40, 0, 1)
```

Hungarian assignment chooses the best one-to-one pairing.

## File score

Relevant paths receive file F1.

The new branch additionally tries to recover unambiguous
repository-relative path variants.

## 11. Why the reward is not simply "accuracy"

The score is not a binary answer key.

For example, a response can receive partial semantic action credit.

Therefore:

``` text
reward = task-specific evaluator score
```

not:

``` text
reward = real-world security accuracy
```

A reward of 0.8 means the output matched the evaluator strongly under
this scoring system.

It does not prove the model found a genuinely exploitable vulnerability.

## 12. Group advantage

After five rewards are obtained:

``` text
mean = group reward mean
std  = population standard deviation
```

The new branch computes:

``` text
A_i =
    (R_i - mean)
    / max(std, 0.05)
```

The standard branch uses:

``` text
A_i =
    (R_i - mean)
    / (std + 1e-6)
```

The new floor prevents tiny differences from being magnified too
aggressively.

If all responses have effectively identical rewards, there is no useful
relative ranking and the update is skipped.

## 13. From reward to policy update

Once the advantages exist, the optimizer does not simply increase the
reward number.

It changes the probability of generated tokens.

For a response:

``` text
old_logp
reference_logp
current_logp
```

are used to construct the policy objective.

The policy ratio is:

``` text
ratio = exp(current_logp - old_logp)
```

The PPO-style clipping is:

``` text
min(
    ratio * advantage,
    clip(ratio, 0.8, 1.2) * advantage
)
```

This limits excessively large policy changes.

## 14. Why the reference model exists

The frozen reference represents the original SFT behavior.

The current policy is encouraged to improve reward while remaining
reasonably close to the SFT distribution.

The KL-like term is:

``` text
sampled_KL = exp(reference_logp - current_logp)
             - (reference_logp - current_logp)
             - 1
```

with coefficient:

``` text
0.02
```

Conceptually:

``` text
Reward pressure
      +
Stay near useful SFT behavior
      =
Controlled RL update
```

The reference does not know whether an answer is secure. It only
regularizes policy movement.

## 15. Why the reasoning tokens are updated

The loss is computed over the completion tokens.

The completion contains:

``` text
REASONING tokens
+
FINAL tokens
```

The same scalar advantage is associated with the completion.

Therefore a high-reward response can increase the probability of both
its reasoning and final tokens.

This does not mean every reasoning token is individually correct.

It means:

> the model receives outcome-based credit for the complete sequence that
> led to a high evaluator reward.

That distinction should be retained in project documentation and
presentations.

## 16. Important token-length effect

The loss is normalized by the total number of completion tokens.

Therefore a response with more generated tokens can contribute more
token-level terms.

This means the five responses are not necessarily equivalent to:

``` text
mean(loss_response_1, ..., loss_response_5)
```

Instead, token counts influence the aggregate.

This matters especially in the reasoning branch because reasoning
increases the expected completion length.

## 17. Context budget is part of the RL problem

The model cannot reason over source code it never sees.

The new branch therefore raises the input budget:

``` text
8192 tokens
```

If the complete repository is larger, the builder truncates.

The allocation tries to distribute source budget across files and
retains both head and tail portions of oversized files.

Coverage is logged.

This is better than silently dropping arbitrary files, but it is still
not semantic retrieval.

## 18. Disposable capacity trial

This is an engineering safeguard, not a learning phase.

The current defaults use:

``` text
TRIAL_MODE = True
```

The notebook chooses three of the longest training prompts.

For each:

``` text
5 rollouts
```

are generated and a full optimizer update is attempted.

The purpose is to test:

``` text
Can this workload fit through generation + reward + backward + optimizer?
```

A model generating an answer successfully does not prove that the
backward pass will fit in GPU memory.

## 19. Why the trial is disposable

The trial temporarily modifies:

``` text
policy weights
optimizer state
AMP scaler state
```

The notebook saves/restores these states.

No trained adapter is exported from the trial.

After the trial:

``` text
restart runtime
```

before beginning the real run.

This is the cleanest way to prevent temporary capacity-test state from
contaminating the actual experiment.

## 20. Synthetic advantages in the trial

A special case exists only to test backward memory.

If the five actual rewards have effectively zero variance, the optimizer
would normally have no useful relative signal.

The trial may temporarily substitute:

``` text
0, 1, 2, 3, 4
```

as artificial advantages.

This is explicitly marked in the logs.

It is not part of real training.

The actual reward remains the true reward from the evaluator.

## 21. Real training after the trial

After a successful capacity trial:

``` text
restart
TRIAL_MODE = False
REASONING_ENABLED = True
```

Then the actual loop begins.

For each of 80 training scenarios:

``` text
context
  -> 5 responses
  -> final extraction
  -> reward
  -> advantage
  -> GRPO-style loss
  -> backward
  -> AdamW update
```

At validation intervals:

``` text
current policy
   -> validation scenarios
   -> mean validation reward
   -> checkpoint ranking
```

## 22. Checkpoint strategy

The project does not simply save:

``` text
final_model.zip
```

Instead it ranks eligible policy states using validation reward.

Only:

``` text
best_adapter.zip
second_best_adapter.zip
```

are retained.

This reduces storage and prevents assuming that the last training state
is the best.

The final in-memory policy can still differ from the saved best adapter.

## 23. How to interpret best vs final

This distinction is important.

Suppose:

``` text
Episode 10 -> validation 0.42
Episode 20 -> validation 0.51  <- best
Episode 30 -> validation 0.48
Episode 40 -> validation 0.45
```

The best adapter corresponds to episode 20.

The last policy corresponds to episode 40.

Therefore:

``` text
best adapter != necessarily final in-memory policy
```

The notebooks explicitly preserve ranking metadata to make this
distinction visible.

## 24. What happens to the test set

The notebooks evaluate the last in-memory policy using a test report.

This is not the same as saying the saved best adapter is the test model.

If the research question is:

> "How well did the best validation checkpoint generalize?"

then the best adapter should be loaded and evaluated separately.

The existing `test_last_policy.json` should therefore be interpreted
literally: it evaluates the last policy state.

## 25. How to compare the standard and new approaches fairly

A meaningful comparison should hold constant:

-   base model
-   SFT adapters
-   scenario IDs
-   random split seed
-   reward implementation
-   evaluator references
-   rollout count
-   learning rate
-   validation split
-   decoding settings where possible

Then compare the intended changes.

A good experimental table is:

  Dimension             Standard   New
  --------------------- ---------- ------------------
  Input                 6144       8192
  Output                384        768
  Reasoning             No         Yes
  Final boundary        No         Yes
  Path recovery         Basic      Repository-aware
  Advantage floor       No         0.05
  Capacity trial        No         Yes
  Reward                Same       Same
  Core GRPO objective   Same       Same

If several changes are introduced simultaneously, an improvement cannot
be attributed to reasoning alone.

This is a major experimental-design issue.

## 26. Better ablation plan

To isolate the effect of reasoning, the strongest next experiment is not
simply:

``` text
standard vs new
```

because the new branch changes several variables.

Instead use:

### A --- Standard baseline

``` text
6144
384
no reasoning
no path recovery
no advantage floor
```

### B --- Context only

``` text
8192
384
no reasoning
```

### C --- Context + reasoning

``` text
8192
768
reasoning enabled
```

### D --- Context + reasoning + engineering improvements

``` text
8192
768
reasoning
path recovery
advantage floor
```

Then:

``` text
A -> B
```

measures additional context.

``` text
B -> C
```

measures the reasoning/output-budget intervention.

``` text
C -> D
```

measures the additional scorer/training stabilizations.

This is much stronger evidence than claiming the entire improvement came
from reasoning.

## 27. What should be monitored during training

The most useful metrics are not just mean reward.

Monitor:

### Reward

``` text
mean reward
median reward
action F1
file F1
```

### Group signal

``` text
reward std
advantage mean/std
number of skipped zero-variance groups
```

### Generation

``` text
completion length
FINAL extraction success
EOS success
reasoning length
```

### Context

``` text
input token count
retained source fraction
omitted ranges
files fully/partially retained
```

### Optimization

``` text
loss
sampled KL
policy ratio
clip fraction
gradient norm
AMP scale
overflow/retry count
```

### Validation

``` text
validation mean reward
best checkpoint
second-best checkpoint
```

These metrics distinguish:

``` text
model quality problem
vs
parser problem
vs
context problem
vs
GPU/numerical problem
```

## 28. Diagnosing common failure modes

## Case 1: All rewards are zero

First inspect:

``` text
validation_before.json
episodes.jsonl
```

Likely causes include:

-   strict extraction failure;
-   malformed FINAL section;
-   incomplete arrays;
-   generation ending incorrectly;
-   wrong action/defense field;
-   paths that cannot be resolved.

Do not immediately assume RL is failing.

## Case 2: Reward is nonzero but validation does not improve

Possible causes:

-   weak reward discrimination;
-   noisy semantic matching;
-   insufficient training steps;
-   context omissions;
-   reward hacking;
-   SFT already near the evaluator ceiling.

## Case 3: File score improves but action score stays zero

The combined reward still gives the file component 20% weight.

Therefore the model can receive positive learning signal from files even
when actions are wrong.

This is expected from the current reward design.

## Case 4: Reasoning becomes longer without better reward

The reasoning itself is not directly rewarded for quality.

More tokens do not imply better reasoning.

The useful metric is whether:

``` text
reasoning-enabled model
```

improves final action/file reward or another independently validated
quality measure.

## Case 5: GPU OOM

Do not assume generation fitting means training fits.

Check the capacity trial first.

If it fails:

-   reduce input/output budgets;
-   restart;
-   rerun the trial;
-   inspect the longest prompts and coverage.

## 29. Fundamental limitations of the current reward

The reward has a deliberate but imperfect structure.

It assumes metadata actions are exhaustive.

That means the model is rewarded for matching the reference, not for
discovering a defensible alternative that the metadata omitted.

Semantic embeddings can also produce false partial matches.

For short security labels, cosine similarity may not capture the actual
mechanism.

Therefore reward quality is itself a research dependency.

## 30. What a stronger future evaluator would look like

The next generation of the project could use multiple evidence levels:

``` text
Level 1: exact/semantic action matching
Level 2: relevant-file matching
Level 3: code evidence verification
Level 4: mechanism consistency
Level 5: executable/static-analysis confirmation
```

Then the reward could distinguish:

``` text
"model named the right attack"
```

from:

``` text
"model identified the right attack and correctly explained the vulnerable mechanism"
```

That would address the biggest conceptual limitation of the current
reasoning branch.

## 31. What the current project can legitimately claim

A defensible claim is:

> The system uses role-specific SFT adapters and a custom GRPO-style
> reinforcement-learning loop to optimize Red and Blue security
> predictions against a deterministic metadata-based evaluator. The
> newer branch provides more repository context and permits a short
> code-grounded reasoning section, while isolating the structured FINAL
> answer for scoring. Reasoning receives indirect outcome-based
> policy-gradient credit because the whole completion is optimized, but
> reasoning is not independently verified.

A claim that should **not** be made without further evidence is:

> The RL system learns verified security reasoning.

The current implementation does not establish that.

## 32. End-to-end process

``` text
                SFT
                 |
       +---------+---------+
       |                   |
   Red adapter          Blue adapter
       |                   |
       v                   v
  Context builder      Context builder
       |                   |
       v                   v
  5 Red rollouts       5 Blue rollouts
       |                   |
       v                   v
  FINAL extraction     FINAL extraction
       |                   |
       v                   v
  Reward evaluator     Reward evaluator
       |                   |
       v                   v
  Group advantages     Group advantages
       |                   |
       v                   v
  GRPO-style update    GRPO-style update
       |                   |
       v                   v
  Validation           Validation
       |                   |
       +---------+---------+
                 |
                 v
       Best / second-best LoRA
```

## 33. Recommended experiment sequence

### Experiment 1 --- Standard baseline

Run the existing standard Red and Blue notebooks.

Record:

-   baseline validation reward
-   action F1
-   file F1
-   extraction rate
-   average completion length
-   GPU memory behavior

### Experiment 2 --- New capacity trial

Run the context-and-reasoning notebooks with:

``` text
TRIAL_MODE=True
REASONING_ENABLED=True
8192 input
768 output
```

Confirm the backward path fits.

### Experiment 3 --- Full context + reasoning

Restart and run:

``` text
TRIAL_MODE=False
REASONING_ENABLED=True
```

Record the same metrics.

### Experiment 4 --- Ablation

Run context-only without reasoning.

This separates:

``` text
more context
```

from:

``` text
reasoning-enabled generation
```

### Experiment 5 --- Reasoning quality audit

Manually inspect a representative set of reasoning outputs.

Check whether explanations:

-   cite the correct files;
-   identify the actual mechanism;
-   agree with the final action;
-   avoid unsupported claims;
-   remain grounded in supplied code.

This is currently a qualitative audit because no reasoning verifier
exists.

## 34. Final conceptual model

The complete system can be understood as:

``` text
SFT
  teaches the task
       |
       v
Context
  supplies repository evidence
       |
       v
Reasoning-enabled policy
  proposes explanation + answer
       |
       v
FINAL parser
  isolates the actual prediction
       |
       v
Deterministic evaluator
  compares prediction to metadata
       |
       v
Reward
  80% actions + 20% files
       |
       v
Group-relative advantage
  says which sampled response was better
       |
       v
GRPO-style update
  shifts policy toward better outcomes
       |
       v
Validation
  identifies stronger policy states
       |
       v
LoRA adapters
```

The key idea is therefore not:

> "Give the model reasoning tokens and it automatically becomes a better
> reasoner."

The actual mechanism is:

> **Give the model enough code context to form an answer, allow it to
> express a short explanation, isolate the final prediction for reliable
> scoring, and use outcome-based group-relative RL to reinforce complete
> responses that score better under the evaluator.**

That is the technically accurate description of the current new RL
approach.
