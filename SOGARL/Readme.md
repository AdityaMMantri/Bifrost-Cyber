Here's the full locked SOGARL RL pipeline, straight from your 3 docs — one complete episode, step by step.

## Core loop (1 episode = 1 scenario)

**1. Load scenario** — `scenario_loader.py` pulls scenario folder (metadata.json, scenario.md, codebase/).

**2. Extract safe scenario.md sections only** — Only these sections go to the models: Description, Overview, Business Domain, Technology Stack, Architecture, Business/Normal Workflow, Trust Boundary, Security Model, Canonical Security Boundary. Anything that reveals the vuln/fix is excluded.

**3. Load metadata internally (Oracle-only)** — `metadata_loader.py` loads `valid_attacks`, `valid_defenses`, root cause, relevant files. **Models never see this. Oracle does.**

**4. Resolve files** — `path_resolver.py` grabs all metadata-listed relevant files + 2–5 noise files. No hardcoded language filtering — works on any file type.

**5. Load source** — `code_loader.py` reads the selected files into `ScenarioContext`.

**6. Build Red Turn-1 prompt** — Repository + safe scenario context. Task: find the strongest valid attack, with root cause, relevant files, reasoning, exploitability — no invented files/APIs.

**7. Red generates G=8 attack candidates** (A1–A8).

**8. Oracle scores all 8** — deterministic checks (label match, file overlap, format) + semantic judging (reasoning quality, root-cause correctness, hallucination penalty) → `Oracle Reward` per candidate.

**9. Compute Red Oracle advantages** — group-relative:
`A_oracle(i) = (R_i − mean(R)) / (std(R) + ε)`

**10. Select Attack_best** — highest-scoring candidate. Only this one goes to Blue; the other 7 stay in Red's GRPO group only.

**11. Confidence gate** — if Attack_best is reliable (category-relative threshold), Blue gets it as context. If not, Blue gets scenario + repo only (no attack hint). This only affects the Turn-1→Turn-2 handoff.

**12. Build Blue Turn-2 prompt** — Repository + scenario + (Attack_best if gate passes). Task: propose the strongest practical defense — strategy, files to change, implementation logic, why it closes root cause, trade-offs.

**13. Blue generates G=8 defense candidates** (D1–D8).

**14. Oracle scores all 8** → Blue Oracle rewards.

**15. Compute Blue Oracle advantages** — same group-relative formula, own group.

**16. Select Blue Top-K=3** — the 3 highest-scoring defenses go on to adversarial testing (e.g. D1, D3, D2).

**17. Turn 3 — Red challenges each Top-K defense** — for each of the 3, build a fresh Red prompt (repo + scenario + that one Blue defense). Task: try to break it — state whether a real exploitable vuln remains, or explicitly confirm the defense holds. No inventing vulnerabilities.

**18. Oracle verifies each challenge** against ground truth → determines interaction outcome per the locked interaction matrix:

| Red claim | Ground truth | Red interaction | Blue interaction |
|---|---|---|---|
| Attack X exists | X valid, still exploitable | +1 | −1 |
| Attack X exists | X fabricated, nothing real remains | −1 | +1 |
| Attack X exists | X fabricated, different real vuln Y remains | 0 | −1 |
| No attack remains | Genuinely closed | +1 | +1 |
| No attack remains | Real vuln remains | −1 | −1 |

Key rule: **Blue's interaction reward tracks the actual security state, not whether Red found it.** A bad Blue defense stays −1 even if Red misses the real hole.

**19. Normalize interaction rewards** — only across the 3 tested defenses:
`A_interaction(i) = (I_i − mean(I)) / (std(I) + ε)`

**20. Untested D4–D8 get A_interaction = 0** — zero = "no evidence," not "bad."

**21. Blue advantage fusion:**
`A_final = α·A_oracle + β·A_interaction` (α=0.8, β=0.2, locked defaults — you're using an adaptive β ramp 0→0.2)

**22. Red GRPO update** — uses A_oracle_red (Turn-1) only, across all 8 candidates. **One update per episode.**

**23. Blue GRPO update** — uses A_final across all 8 candidates. **One update per episode.**

**24. Turn-3 Red rewards are NOT a second Red GRPO update in v1** — logged for metrics/eval/replay only. Reason: Turn 1 (attack discovery) and Turn 3 (defense verification) are different generation objectives; a dedicated Turn-3 Red objective is a documented future extension.

**25. Log episode** to replay buffer (`episodes.jsonl`) + metrics (`red_metrics.csv`, `blue_metrics.csv`).

**26. Update weakness stats** — per-category failure rates.

**27. Checkpoint if due** — LoRA adapters + optimizer state + RNG state saved (base model not re-copied).

**28. Weakness sampler picks next scenario** — 70% from high-failure categories (once ≥20 logged episodes/category), 30% random — and the loop repeats.

---

## What's fixed vs. what trains

| | Turn 1 (Red) | Turn 2 (Blue) | Turn 3 (Red) |
|---|---|---|---|
| Rollouts | G=8 | G=8 → Top-K=3 challenged | 3 challenges |
| Reward source | Oracle only | Oracle + Interaction (fused) | Interaction matrix |
| GRPO update? | Yes — 1x/episode | Yes — 1x/episode | No (metrics only, v1) |

## Ground-truth isolation (locked boundary)

- **Model context:** `scenario.md` (safe sections only) + `codebase/`
- **Oracle-only context:** `metadata.json` — valid_attacks, valid_defenses, root cause, security boundaries
- SFT artifacts (`red_sft.json`, `blue_sft.json`) are **not loaded by RL at all** — RL runs purely off scenario.md + codebase + metadata.

## Build/file structure (already in your memory, confirmed by the file map doc)

```
config.py
data/ → models.py, scenario_loader.py, metadata_loader.py, path_resolver.py, code_loader.py
utils/ → file_utils, random_utils, context_estimator, logger, checkpoint, seed, device
generation/ → prompt_builder.py, generator.py
oracle/ → deterministic_checks.py, interaction_checker.py, oracle.py
sampling/ → weakness_sampler.py
logging/ → replay_buffer.py, metrics_logger.py
rl/ → reward_manager.py, grpo.py, episode_manager.py
scripts/ → test_oracle.py → test_episode.py → validate_dataset.py → evaluate.py → train.py
```

Pre-training gate before any real run: `validate_dataset.py` → `test_oracle.py` → `test_episode.py` (1 full Red→Blue→Red-challenge→Oracle→reward-fusion→GRPO cycle) → small run (1–3 scenarios, 1 epoch) → only then `train.py` full run.

Given your file map says you still need `generator.py` → `episode_manager.py` → `grpo.py` → `replay_buffer.py` → `train.py`, this 28-step episode spec above is exactly what `episode_manager.py` needs to implement — want me to draft that next, or the reward_manager/GRPO math first since episode_manager depends on it?