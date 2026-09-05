"""
SOGARL Semantic Oracle Judge
============================

Oracle-side semantic evaluation for SOGARL.

Responsibilities
----------------
1. Semantically evaluate Red Turn-1 attack candidates.
2. Semantically evaluate Blue Turn-2 defense candidates.
3. Semantically evaluate Red Turn-3 challenges against Blue defenses.
4. Return structured JSON-compatible Oracle evidence.
5. Use hidden scenario metadata only inside the Oracle.
6. Reuse the already-loaded SOGARL shared base model.
7. Disable Red/Blue LoRA adapters while performing semantic judging.

This module does NOT:

    - build Red prompts
    - build Blue prompts
    - expose hidden metadata to policies
    - perform GRPO
    - calculate GRPO advantages
    - select Attack_best
    - select Blue Top-K
    - update model parameters
    - modify scenario objects

Architecture

                Shared PeftModel
                       |
             +---------+---------+
             |                   |
          Red LoRA           Blue LoRA
             |                   |
          Policy              Policy

                       |
                LoRA disabled
                       |
                       v
                Semantic Oracle

The semantic Oracle is therefore not another base-model load.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import torch

from configs import config
from src.generation.chat_formatter import format_chat_prompt


# ============================================================================
# SEMANTIC JUDGE
# ============================================================================

class SemanticJudge:
    """
    Semantic evaluator used by the SOGARL Oracle.

    The judge reuses the already-loaded shared model.

    The model is evaluated with all policy LoRA adapters disabled,
    so semantic evaluation is performed by the underlying base model.

    Constructor compatibility
    --------------------------
    Supports both:

        SemanticJudge(
            base_generator=generator,
            logger=logger,
        )

    and:

        SemanticJudge(
            model=model,
            tokenizer=tokenizer,
            device=device,
            logger=logger,
        )

    The second form is what the current Oracle uses.
    """

    # ==================================================================
    # INITIALIZATION
    # ==================================================================

    def __init__(
        self,
        base_generator=None,
        logger=None,
        model=None,
        tokenizer=None,
        device=None,
    ) -> None:

        self.base_generator = base_generator
        self.logger = logger
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # --------------------------------------------------------------
        # If a Generator was supplied, obtain its underlying shared model.
        # --------------------------------------------------------------

        if self.base_generator is not None:

            if self.model is None:
                self.model = self._unwrap_generator_model(
                    self.base_generator
                )

            if self.tokenizer is None:
                self.tokenizer = self.base_generator.get_tokenizer()

            if self.device is None:
                self.device = self.base_generator.get_device()

        # --------------------------------------------------------------
        # Final validation.
        # --------------------------------------------------------------

        if self.model is None:
            raise ValueError(
                "SemanticJudge requires an already-loaded "
                "SOGARL model."
            )

        if self.tokenizer is None:
            raise ValueError(
                "SemanticJudge requires a tokenizer."
            )

        if self.device is None:
            raise ValueError(
                "SemanticJudge requires a model device."
            )

    # ==================================================================
    # PUBLIC CALL INTERFACE
    # ==================================================================

    def __call__(
        self,
        response: str,
        scenario,
        candidate_type: str,
        attack_response: Optional[str] = None,
        blue_defense: Optional[str] = None,
        judgments: int = 2,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:

        candidate_type = str(candidate_type).strip().lower()

        if candidate_type == "attack":
            return self.score_attack(
                response=response,
                scenario=scenario,
                judgments=judgments,
                temperature=temperature,
            )

        if candidate_type == "defense":
            return self.score_defense(
                response=response,
                scenario=scenario,
                attack_response=attack_response,
                judgments=judgments,
                temperature=temperature,
            )

        if candidate_type == "interaction":
            return self.evaluate_interaction(
                response=response,
                scenario=scenario,
                blue_defense=blue_defense or "",
                judgments=judgments,
                temperature=temperature,
            )

        raise ValueError(
            f"Unsupported semantic candidate type: "
            f"{candidate_type}"
        )

    # ==================================================================
    # ATTACK SCORING
    # ==================================================================

    def score_attack(
        self,
        response: str,
        scenario,
        judgments: int = 2,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:

        metadata = self._oracle_metadata(scenario)

        prompt = self._build_attack_prompt(
            response=response,
            metadata=metadata,
        )

        results = self._sample_judgments(
            prompt=prompt,
            judgments=judgments,
            temperature=temperature,
        )

        return self._aggregate_candidate_judgments(results)

    # ==================================================================
    # DEFENSE SCORING
    # ==================================================================

    def score_defense(
        self,
        response: str,
        scenario,
        attack_response: Optional[str] = None,
        judgments: int = 2,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:

        metadata = self._oracle_metadata(scenario)

        prompt = self._build_defense_prompt(
            response=response,
            attack_response=attack_response,
            metadata=metadata,
        )

        results = self._sample_judgments(
            prompt=prompt,
            judgments=judgments,
            temperature=temperature,
        )

        return self._aggregate_candidate_judgments(results)

    # ==================================================================
    # TURN-3 INTERACTION
    # ==================================================================

    def evaluate_interaction(
        self,
        response: str,
        scenario,
        blue_defense: str,
        judgments: int = 2,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:

        metadata = self._oracle_metadata(scenario)

        prompt = self._build_interaction_prompt(
            response=response,
            blue_defense=blue_defense,
            metadata=metadata,
        )

        results = self._sample_judgments(
            prompt=prompt,
            judgments=judgments,
            temperature=temperature,
        )

        aggregated = self._aggregate_interaction_judgments(
            results
        )

        return self._validate_interaction_result(
            aggregated,
            metadata,
        )

    # ==================================================================
    # HIDDEN ORACLE METADATA
    # ==================================================================

    def _oracle_metadata(
        self,
        scenario,
    ) -> Dict[str, Any]:

        metadata = getattr(
            scenario,
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        vulnerabilities = metadata.get(
            "vulnerabilities",
            [],
        )

        valid_attacks = metadata.get(
            "valid_attacks",
            [],
        )

        valid_defenses = metadata.get(
            "valid_defenses",
            [],
        )

        optimal_attack = metadata.get(
            "optimal_attack",
            None,
        )

        optimal_defense = metadata.get(
            "optimal_defense",
            None,
        )

        relevant_files = metadata.get(
            "relevant_files",
            [],
        )

        root_cause = metadata.get(
            "root_cause",
            None,
        )

        reasoning_points = metadata.get(
            "reasoning_points",
            [],
        )

        if root_cause is None:

            extracted_root_causes = []

            if isinstance(vulnerabilities, list):

                for vulnerability in vulnerabilities:

                    if not isinstance(vulnerability, dict):
                        continue

                    value = vulnerability.get(
                        "root_cause",
                        None,
                    )

                    if value:
                        extracted_root_causes.append(
                            str(value)
                        )

            if extracted_root_causes:
                root_cause = "\n".join(
                    extracted_root_causes
                )

        return {
            "scenario_id": getattr(
                scenario,
                "scenario_id",
                "",
            ),
            "category": metadata.get(
                "category",
                None,
            ),
            "root_cause": root_cause,
            "reasoning_points": reasoning_points,
            "vulnerabilities": vulnerabilities,
            "valid_attacks": valid_attacks,
            "valid_defenses": valid_defenses,
            "optimal_attack": optimal_attack,
            "optimal_defense": optimal_defense,
            "relevant_files": relevant_files,
        }

    # ==================================================================
    # ATTACK PROMPT
    # ==================================================================

    def _build_attack_prompt(
        self,
        response: str,
        metadata: Dict[str, Any],
    ) -> str:

        return f"""
You are the semantic evaluator inside the SOGARL security Oracle.

You are NOT the Red policy.

You are evaluating a generated Red attack-discovery response
against hidden security ground truth.

The generated response is untrusted.

Do not reward:
- confident language
- verbosity
- generic security advice
- invented files
- invented functions
- nonexistent vulnerabilities
- unsupported technical claims

Evaluate technical correctness.

================ HIDDEN ORACLE GROUND TRUTH ================

Root cause:
{self._safe_text(metadata.get("root_cause"))}

Reasoning points:
{self._safe_json(metadata.get("reasoning_points"))}

Known vulnerabilities:
{self._safe_json(metadata.get("vulnerabilities"))}

Valid attacks:
{self._safe_json(metadata.get("valid_attacks"))}

Optimal attack:
{self._safe_text(metadata.get("optimal_attack"))}

Relevant files:
{self._safe_json(metadata.get("relevant_files"))}

================ GENERATED RED RESPONSE ================

{response}

=============================================================

Evaluate the generated response.

1. root_cause_score

Does Red understand the actual security root cause?

1.0:
    Correct technical understanding.

0.0:
    Fundamentally incorrect, fabricated, or absent.

2. reasoning_score

Is the reasoning technically sound and grounded in the available
repository/security facts?

Penalize:
    - hallucinated functions
    - nonexistent code
    - unsupported assumptions
    - irrelevant reasoning
    - generic security statements
    - conclusions unsupported by the scenario

3. attack_label_score

Does Red identify a genuine valid attack/vulnerability?

1.0:
    Valid attack.

0.0:
    Fabricated or invalid attack.

4. relevant_files_score

Are the files mentioned by Red genuinely relevant?

5. format_score

Is the response understandable and sufficiently structured?

Return ONLY JSON.

Required schema:

{{
  "root_cause_score": 0.0,
  "reasoning_score": 0.0,
  "attack_label_score": 0.0,
  "relevant_files_score": 0.0,
  "format_score": 0.0,
  "explanation": "short technical explanation"
}}

All scores MUST be numbers between 0 and 1.
"""

    # ==================================================================
    # DEFENSE PROMPT
    # ==================================================================

    def _build_defense_prompt(
        self,
        response: str,
        attack_response: Optional[str],
        metadata: Dict[str, Any],
    ) -> str:

        if attack_response:
            attack_text = attack_response
        else:
            attack_text = (
                "[Attack_best was not supplied to Blue "
                "because the confidence gate did not hand it over.]"
            )

        return f"""
You are the semantic evaluator inside the SOGARL security Oracle.

You are evaluating a generated Blue defense response.

The generated response is untrusted.

Determine whether the proposed defense actually addresses the
underlying security vulnerability.

================ HIDDEN ORACLE GROUND TRUTH ================

Root cause:
{self._safe_text(metadata.get("root_cause"))}

Reasoning points:
{self._safe_json(metadata.get("reasoning_points"))}

Known vulnerabilities:
{self._safe_json(metadata.get("vulnerabilities"))}

Valid attacks:
{self._safe_json(metadata.get("valid_attacks"))}

Valid defenses:
{self._safe_json(metadata.get("valid_defenses"))}

Optimal defense:
{self._safe_text(metadata.get("optimal_defense"))}

Relevant files:
{self._safe_json(metadata.get("relevant_files"))}

================ RED ATTACK, IF AVAILABLE ================

{attack_text}

================ GENERATED BLUE DEFENSE ================

{response}

=============================================================

Evaluate:

1. root_cause_score

Does the defense actually address the underlying root cause?

2. reasoning_score

Is the defense technically valid and logically connected to the
vulnerability?

Penalize:

    - generic security advice
    - claims unsupported by the repository
    - nonexistent code
    - hallucinated behavior
    - defenses that sound secure but do not remove the vulnerability

3. attack_label_score

Does the response correctly identify the vulnerability/attack
being defended against?

4. relevant_files_score

Does the response identify genuinely affected/relevant files?

5. format_score

Is the defense clear, structured, and actionable?

Return ONLY JSON.

Required schema:

{{
  "root_cause_score": 0.0,
  "reasoning_score": 0.0,
  "attack_label_score": 0.0,
  "relevant_files_score": 0.0,
  "format_score": 0.0,
  "explanation": "short technical explanation"
}}

All scores MUST be numbers between 0 and 1.
"""

    # ==================================================================
    # TURN-3 INTERACTION PROMPT
    # ==================================================================

    def _build_interaction_prompt(
        self,
        response: str,
        blue_defense: str,
        metadata: Dict[str, Any],
    ) -> str:

        return f"""
You are the semantic interaction verifier inside SOGARL.

You are evaluating a Red Turn-3 adversarial challenge against
a Blue defense.

Your job is to independently determine the factual security state
AFTER the proposed Blue defense.

Do NOT simply trust Red's statement.

Use the hidden ground truth and technical reasoning.

================ HIDDEN ORACLE GROUND TRUTH ================

Root cause:
{self._safe_text(metadata.get("root_cause"))}

Known vulnerabilities:
{self._safe_json(metadata.get("vulnerabilities"))}

Valid attacks:
{self._safe_json(metadata.get("valid_attacks"))}

Valid defenses:
{self._safe_json(metadata.get("valid_defenses"))}

Optimal defense:
{self._safe_text(metadata.get("optimal_defense"))}

Relevant files:
{self._safe_json(metadata.get("relevant_files"))}

================ BLUE DEFENSE ================

{blue_defense}

================ RED TURN-3 CHALLENGE ================

{response}

=============================================================

Determine ALL of the following:

1. Does Red claim that an attack exists?

2. What attack does Red claim, if any?

3. Is Red's claimed attack a genuine valid attack?

4. Does the claimed attack remain exploitable AFTER the Blue defense?

5. Does ANY genuine vulnerability remain AFTER the Blue defense?

6. List every genuine remaining attack.

7. If Red claims that no attack remains, independently determine
whether that claim is correct.

IMPORTANT:

A fabricated attack means:

    claimed_attack_valid = false

If the named attack is valid but the Blue defense closes it:

    claimed_attack_valid = true
    claimed_attack_remains = false

If another genuine attack remains:

    real_attack_remains = true
    remaining_attacks = [...]

If no genuine attack remains:

    real_attack_remains = false
    remaining_attacks = []
    no_attack_verified = true

Do not confuse:

    "Red named the correct attack"

with:

    "The attack remains exploitable."

Return ONLY JSON.

Required schema:

{{
  "red_claim_type": "attack",
  "claimed_attack": null,
  "claimed_attack_valid": false,
  "claimed_attack_remains": false,
  "no_attack_verified": false,
  "real_attack_remains": false,
  "remaining_attacks": [],
  "verified_attack": null,
  "explanation": "short technical explanation"
}}

Rules:

red_claim_type MUST be exactly:

    "attack"

or:

    "no_attack"

All boolean fields MUST be true or false.

remaining_attacks MUST always be a JSON list.

Do not return Markdown.
Do not return additional text.
"""

    # ==================================================================
    # JUDGMENT SAMPLING
    # ==================================================================

    def _sample_judgments(
        self,
        prompt: str,
        judgments: int,
        temperature: float,
    ) -> List[Dict[str, Any]]:

        try:
            count = int(judgments)
        except Exception:
            count = 1

        count = max(
            1,
            count,
        )

        results = []

        for index in range(count):

            try:

                raw = self._generate_base(
                    prompt=prompt,
                    temperature=temperature,
                )

                parsed = self._parse_json(raw)

                if parsed is not None:

                    results.append(parsed)

                else:

                    self._log(
                        "Semantic judgment "
                        f"{index + 1} returned invalid JSON."
                    )

            except Exception as exc:

                self._log(
                    "Semantic judgment "
                    f"{index + 1} failed: {exc}"
                )

        return results

    # ==================================================================
    # BASE MODEL GENERATION
    # ==================================================================

    @torch.inference_mode()
    def _generate_base(
        self,
        prompt: str,
        temperature: float,
    ) -> str:

        model = self.model
        tokenizer = self.tokenizer

        formatted_prompt = format_chat_prompt(
            tokenizer,
            prompt,
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(
                config.MAX_INPUT_TOKENS
            ),
            padding=True,
            return_attention_mask=True,
            add_special_tokens=False,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        was_training = bool(
            getattr(
                model,
                "training",
                False,
            )
        )

        previous_adapter = self._get_active_adapter(
            model
        )

        model.eval()

        try:

            disable_adapter = getattr(
                model,
                "disable_adapter",
                None,
            )

            if callable(disable_adapter):

                with disable_adapter():

                    output = self._raw_generate(
                        model=model,
                        inputs=inputs,
                        temperature=temperature,
                    )

            else:

                output = self._raw_generate(
                    model=model,
                    inputs=inputs,
                    temperature=temperature,
                )

        finally:

            self._restore_adapter(
                model=model,
                adapter=previous_adapter,
            )

            if was_training:

                try:
                    model.train()
                except Exception:
                    pass

            else:

                try:
                    model.eval()
                except Exception:
                    pass

        input_length = inputs["input_ids"].shape[-1]

        generated_tokens = output[:, input_length:]

        text = tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        )

        return text.strip()

    # ==================================================================
    # RAW GENERATION
    # ==================================================================

    def _raw_generate(
        self,
        model,
        inputs,
        temperature: float,
    ):

        temperature = max(
            float(temperature),
            1e-5,
        )

        kwargs = {
            "max_new_tokens": int(
                config.MAX_NEW_TOKENS
            ),
            "do_sample": True,
            "temperature": temperature,
            "top_p": float(
                getattr(
                    config,
                    "TOP_P",
                    0.95,
                )
            ),
            "pad_token_id": self._pad_token_id(),
            "eos_token_id": self._eos_token_id(),
        }

        return model.generate(
            **inputs,
            **kwargs,
        )

    # ==================================================================
    # ADAPTER STATE
    # ==================================================================

    @staticmethod
    def _get_active_adapter(
        model,
    ):
        return getattr(
            model,
            "active_adapter",
            None,
        )

    @staticmethod
    def _restore_adapter(
        model,
        adapter,
    ) -> None:

        if adapter is None:
            return

        setter = getattr(
            model,
            "set_adapter",
            None,
        )

        if not callable(setter):
            return

        try:

            setter(adapter)

        except Exception:

            if (
                isinstance(adapter, list)
                and len(adapter) == 1
            ):

                try:
                    setter(adapter[0])
                except Exception:
                    pass

    # ==================================================================
    # TOKENIZER HELPERS
    # ==================================================================

    def _pad_token_id(
        self,
    ) -> Optional[int]:

        value = getattr(
            self.tokenizer,
            "pad_token_id",
            None,
        )

        if value is not None:
            return value

        return getattr(
            self.tokenizer,
            "eos_token_id",
            None,
        )

    def _eos_token_id(
        self,
    ) -> Optional[int]:

        return getattr(
            self.tokenizer,
            "eos_token_id",
            None,
        )

    # ==================================================================
    # CANDIDATE AGGREGATION
    # ==================================================================

    def _aggregate_candidate_judgments(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        score_keys = [
            "root_cause_score",
            "reasoning_score",
            "attack_label_score",
            "relevant_files_score",
            "format_score",
        ]

        if not results:

            return {
                "root_cause_score": None,
                "reasoning_score": None,
                "attack_label_score": None,
                "relevant_files_score": None,
                "format_score": None,
                "semantic_score": None,
                "semantic_available": False,
            }

        output = {}

        for key in score_keys:

            values = []

            for result in results:

                value = self._score(
                    result.get(
                        key,
                        None,
                    )
                )

                if value is not None:
                    values.append(value)

            if values:

                output[key] = (
                    sum(values)
                    / len(values)
                )

            else:

                output[key] = None

        valid_scores = [
            output[key]
            for key in score_keys
            if output.get(key) is not None
        ]

        if valid_scores:

            output["semantic_score"] = (
                sum(valid_scores)
                / len(valid_scores)
            )

        else:

            output["semantic_score"] = None

        output["semantic_available"] = bool(
            valid_scores
        )

        explanations = []

        for result in results:

            explanation = result.get(
                "explanation",
                "",
            )

            if explanation:

                explanations.append(
                    str(
                        explanation
                    ).strip()
                )

        if explanations:

            output["explanation"] = (
                " | ".join(
                    explanations
                )
            )

        return output

    # ==================================================================
    # INTERACTION AGGREGATION
    # ==================================================================

    def _aggregate_interaction_judgments(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate Turn-3 semantic judgments.

        Boolean fields use MAJORITY VOTING.

        With the recommended configuration:

            ORACLE_JUDGMENTS = 3

        examples:

            True, True, False  -> True
            True, False, True  -> True
            False, False, True -> False
            False, True, False -> False

        An exact tie is returned as None because there is no majority.
        This preserves the safety property that an unsupported factual
        state is never fabricated.

        String/list fields continue to use conservative consensus.
        """

        if not results:
            return {}

        # --------------------------------------------------------------
        # Claim type
        # --------------------------------------------------------------

        claim_type = self._aggregate_claim_type(
            results
        )

        # --------------------------------------------------------------
        # Boolean fields
        #
        # IMPORTANT:
        # This is the stability fix for stochastic Turn 3.
        # --------------------------------------------------------------

        claimed_valid = self._consensus_bool(
            results,
            "claimed_attack_valid",
        )

        claimed_remains = self._consensus_bool(
            results,
            "claimed_attack_remains",
        )

        no_attack_verified = self._consensus_bool(
            results,
            "no_attack_verified",
        )

        real_attack_remains = self._consensus_bool(
            results,
            "real_attack_remains",
        )

        # --------------------------------------------------------------
        # We require every factual boolean to have a majority.
        #
        # With ORACLE_JUDGMENTS=3 this means at least 2/3 agreement.
        # If there is no majority, semantic evidence is incomplete and
        # Oracle is allowed to fall back to deterministic evidence.
        # --------------------------------------------------------------

        if any(
            value is None
            for value in (
                claimed_valid,
                claimed_remains,
                no_attack_verified,
                real_attack_remains,
            )
        ):

            return {}

        # --------------------------------------------------------------
        # String fields
        # --------------------------------------------------------------

        claimed_attack = self._consensus_string(
            results,
            "claimed_attack",
        )

        verified_attack = self._consensus_string(
            results,
            "verified_attack",
        )

        # --------------------------------------------------------------
        # Remaining attacks.
        # --------------------------------------------------------------

        remaining_attacks = (
            self._consensus_remaining_attacks(
                results
            )
        )

        # --------------------------------------------------------------
        # Reconcile final state with remaining_attacks.
        # --------------------------------------------------------------

        if remaining_attacks:

            real_attack_remains = True
            no_attack_verified = False

        elif not real_attack_remains:

            no_attack_verified = True

        # --------------------------------------------------------------
        # If judges agree that Red claims no attack, use canonical form.
        # --------------------------------------------------------------

        if claim_type == "no_attack":

            claimed_attack = None

        # --------------------------------------------------------------
        # Explanation.
        # --------------------------------------------------------------

        explanations = []

        for result in results:

            explanation = result.get(
                "explanation",
                "",
            )

            if explanation:

                explanations.append(
                    str(
                        explanation
                    ).strip()
                )

        explanation = (
            " | ".join(
                explanations
            )
            if explanations
            else
            "Semantic interaction judgment."
        )

        return {
            "red_claim_type": claim_type,
            "claimed_attack": claimed_attack,
            "claimed_attack_valid": claimed_valid,
            "claimed_attack_remains": claimed_remains,
            "no_attack_verified": no_attack_verified,
            "real_attack_remains": real_attack_remains,
            "remaining_attacks": remaining_attacks,
            "verified_attack": verified_attack,
            "explanation": explanation,
            "semantic_available": True,
        }

    # ==================================================================
    # INTERACTION CLAIM TYPE
    # ==================================================================

    def _aggregate_claim_type(
        self,
        results: List[Dict[str, Any]],
    ) -> str:

        claims = []

        for result in results:

            claims.append(
                self._normalize_claim_type(
                    result
                )
            )

        if not claims:

            return "no_attack"

        if all(
            claim == claims[0]
            for claim in claims
        ):

            return claims[0]

        if "attack" in claims:

            return "attack"

        return "no_attack"

    # ==================================================================
    # BOOLEAN MAJORITY
    # ==================================================================

    @staticmethod
    def _consensus_bool(
        results: List[Dict[str, Any]],
        key: str,
    ) -> Optional[bool]:
        """
        Return the majority boolean value.

        This replaces the previous unanimous-consensus behavior.

        Why:
            Turn-3 semantic judgments are sampled with temperature > 0.
            Therefore occasional disagreement is expected.

        Safety:
            - At least one valid boolean is required.
            - A strict majority is required.
            - A tie returns None.
            - Invalid/missing boolean values are ignored.
        """

        if not results:
            return None

        values: List[bool] = []

        for result in results:

            value = result.get(
                key,
                None,
            )

            if isinstance(
                value,
                bool,
            ):

                values.append(value)

        if not values:
            return None

        true_count = sum(
            1
            for value in values
            if value
        )

        false_count = len(values) - true_count

        if true_count > false_count:
            return True

        if false_count > true_count:
            return False

        return None

    # ==================================================================
    # STRING CONSENSUS
    # ==================================================================

    def _consensus_string(
        self,
        results: List[Dict[str, Any]],
        key: str,
    ) -> Optional[str]:

        values = []

        for result in results:

            value = result.get(
                key,
                None,
            )

            if value is None:
                continue

            normalized = self._normalize_text(
                value
            )

            if normalized:

                values.append(
                    (
                        normalized,
                        str(value).strip(),
                    )
                )

        if not values:

            return None

        counts: Dict[str, int] = {}
        originals: Dict[str, str] = {}

        for normalized, original in values:

            counts[normalized] = (
                counts.get(
                    normalized,
                    0,
                )
                + 1
            )

            originals[
                normalized
            ] = original

        winner = max(
            counts,
            key=counts.get,
        )

        required = len(results)

        if counts[winner] < required:

            return None

        return originals[winner]

    # ==================================================================
    # REMAINING ATTACK CONSENSUS
    # ==================================================================

    def _consensus_remaining_attacks(
        self,
        results: List[Dict[str, Any]],
    ) -> List[str]:

        if not results:
            return []

        normalized_sets = []
        display_names: Dict[str, str] = {}

        for result in results:

            attacks = result.get(
                "remaining_attacks",
                [],
            )

            if not isinstance(
                attacks,
                list,
            ):

                attacks = []

            current = set()

            for attack in attacks:

                normalized = self._normalize_text(
                    attack
                )

                if not normalized:
                    continue

                current.add(normalized)

                display_names[
                    normalized
                ] = str(
                    attack
                ).strip()

            normalized_sets.append(
                current
            )

        if not normalized_sets:
            return []

        consensus = normalized_sets[0]

        for current in normalized_sets[1:]:

            consensus = (
                consensus
                & current
            )

        return [
            display_names[attack]
            for attack in consensus
        ]

    # ==================================================================
    # INTERACTION RESULT VALIDATION
    # ==================================================================

    def _validate_interaction_result(
        self,
        result: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not result:
            return {}

        result = dict(result)

        valid_attacks = [
            str(x).strip()
            for x in (
                metadata.get(
                    "valid_attacks"
                )
                or []
            )
            if str(x).strip()
        ]

        claimed = result.get(
            "claimed_attack"
        )

        if (
            claimed
            and not self._label_in(
                claimed,
                valid_attacks,
            )
        ):

            result["claimed_attack_valid"] = False
            result["claimed_attack_remains"] = False

        remaining = result.get(
            "remaining_attacks",
            [],
        )

        if not isinstance(
            remaining,
            list,
        ):

            return {}

        validated = []

        for attack in remaining:

            canonical = self._canonical_label(
                attack,
                valid_attacks,
            )

            if (
                canonical is not None
                and canonical not in validated
            ):

                validated.append(
                    canonical
                )

        result["remaining_attacks"] = validated

        # --------------------------------------------------------------
        # Preserve the semantic factual state.
        # --------------------------------------------------------------

        semantic_real_attack_remains = result.get(
            "real_attack_remains",
            None,
        )

        if isinstance(
            semantic_real_attack_remains,
            bool,
        ):

            real_attack_remains = (
                semantic_real_attack_remains
            )

        else:

            real_attack_remains = bool(
                validated
            )

        if validated:

            real_attack_remains = True

        result["real_attack_remains"] = (
            real_attack_remains
        )

        # --------------------------------------------------------------
        # Preserve explicit no_attack_verified where valid.
        # --------------------------------------------------------------

        semantic_no_attack_verified = result.get(
            "no_attack_verified",
            None,
        )

        if isinstance(
            semantic_no_attack_verified,
            bool,
        ):

            no_attack_verified = (
                semantic_no_attack_verified
            )

        else:

            no_attack_verified = (
                not real_attack_remains
            )

        if validated:

            no_attack_verified = False

        result["no_attack_verified"] = (
            no_attack_verified
        )

        # --------------------------------------------------------------
        # Validate claimed attack.
        # --------------------------------------------------------------

        if (
            result.get(
                "claimed_attack"
            )
            and self._label_in(
                result["claimed_attack"],
                valid_attacks,
            )
        ):

            result["claimed_attack_valid"] = True

            result["claimed_attack_remains"] = any(
                self._normalize_text(
                    result["claimed_attack"]
                )
                == self._normalize_text(
                    x
                )
                for x in validated
            )

        return result

    # ==================================================================
    # LABEL HELPERS
    # ==================================================================

    @staticmethod
    def _label_in(
        value: Any,
        labels: List[str],
    ) -> bool:

        if value is None:
            return False

        normalized = SemanticJudge._normalize_text(
            value
        )

        return any(
            normalized
            == SemanticJudge._normalize_text(label)
            for label in labels
        )

    @staticmethod
    def _canonical_label(
        value: Any,
        labels: List[str],
    ) -> Optional[str]:

        if value is None:
            return None

        for label in labels:

            if (
                SemanticJudge._normalize_text(value)
                == SemanticJudge._normalize_text(label)
            ):

                return label

        return None

    # ==================================================================
    # JSON PARSING
    # ==================================================================

    def _parse_json(
        self,
        text: str,
    ) -> Optional[Dict[str, Any]]:

        if not text:
            return None

        text = str(text).strip()

        fenced = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=(
                re.DOTALL
                |
                re.IGNORECASE
            ),
        )

        if fenced:

            text = fenced.group(1).strip()

        try:

            value = json.loads(text)

            if isinstance(
                value,
                dict,
            ):

                return value

        except Exception:
            pass

        start = text.find("{")

        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False

        for index in range(
            start,
            len(text),
        ):

            char = text[index]

            if escaped:

                escaped = False
                continue

            if (
                char == "\\"
                and in_string
            ):

                escaped = True
                continue

            if char == '"':

                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":

                depth += 1

            elif char == "}":

                depth -= 1

                if depth == 0:

                    candidate = text[
                        start:index + 1
                    ]

                    try:

                        value = json.loads(
                            candidate
                        )

                        if isinstance(
                            value,
                            dict,
                        ):

                            return value

                    except Exception:

                        return None

        return None

    # ==================================================================
    # SCORE NORMALIZATION
    # ==================================================================

    @staticmethod
    def _score(
        value,
    ) -> Optional[float]:

        if value is None:
            return None

        try:

            value = float(value)

        except (
            TypeError,
            ValueError,
        ):

            return None

        if value != value:
            return None

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    # ==================================================================
    # CLAIM NORMALIZATION
    # ==================================================================

    @staticmethod
    def _normalize_claim_type(
        finding: Dict[str, Any],
    ) -> str:

        value = finding.get(
            "red_claim_type",
            finding.get(
                "claim_type",
                "no_attack",
            ),
        )

        value = str(
            value
        ).strip().lower()

        if value in {
            "attack",
            "valid_attack",
            "attack_found",
            "attack_exists",
            "attack exists",
            "vulnerability",
            "vulnerability_exists",
        }:

            return "attack"

        return "no_attack"

    # ==================================================================
    # TEXT NORMALIZATION
    # ==================================================================

    @staticmethod
    def _normalize_text(
        value,
    ) -> str:

        if value is None:
            return ""

        text = str(
            value
        ).strip().lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    # ==================================================================
    # SAFE TEXT SERIALIZATION
    # ==================================================================

    @staticmethod
    def _safe_text(
        value,
    ) -> str:

        if value is None:
            return "unknown"

        if isinstance(
            value,
            (
                dict,
                list,
            ),
        ):

            return SemanticJudge._safe_json(
                value
            )

        return str(value)

    # ==================================================================
    # SAFE JSON SERIALIZATION
    # ==================================================================

    @staticmethod
    def _safe_json(
        value,
    ) -> str:

        try:

            return json.dumps(
                value,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        except Exception:

            return str(value)

    # ==================================================================
    # GENERATOR MODEL UNWRAPPING
    # ==================================================================

    @staticmethod
    def _unwrap_generator_model(
        generator,
    ):

        model = generator.get_model()

        shared_model = getattr(
            model,
            "_shared_model",
            None,
        )

        if shared_model is not None:
            return shared_model

        return model

    # ==================================================================
    # LOGGING
    # ==================================================================

    def _log(
        self,
        message: str,
    ) -> None:

        if self.logger is None:
            return

        try:

            self.logger.warning(
                message
            )

            return

        except Exception:
            pass

        try:

            self.logger.info(
                message
            )

        except Exception:
            pass