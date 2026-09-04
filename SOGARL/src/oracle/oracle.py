"""
oracle.py

Central Oracle for SOGARL.

Responsibilities:
    - Score Red Turn-1 attack candidates.
    - Score Blue Turn-2 defense candidates.
    - Evaluate Red Turn-3 challenges.
    - Combine deterministic and semantic evaluation.
    - Return grounded reward information.

Reward types:

1. Oracle reward
   Continuous value in [0, 1].

   Used for:
       Red Turn 1
       Blue Turn 2

2. Interaction reward
   Discrete value in {-1, 0, +1}.

   Used only for:
       Red Turn 3
       Blue interaction signal

This module does NOT:
    - perform GRPO
    - calculate advantages
    - select Attack_best
    - select Blue Top-K
    - update models
    - modify scenarios
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from configs import config

# ============================================================================
# RESULT OBJECTS
# ============================================================================

@dataclass
class OracleScore:
    """
    Result of normal Oracle evaluation.

    reward:
        Final continuous Oracle reward in [0, 1].

    components:
        Individual scoring components.

    metadata:
        Additional information useful for debugging/logging.
    """

    reward: float
    attack_label_score: float = 0.0
    root_cause_score: float = 0.0
    relevant_files_score: float = 0.0
    reasoning_score: float = 0.0
    format_score: float = 0.0
    deterministic_score: Optional[float] = None
    semantic_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward": self.reward,
            "attack_label_score": self.attack_label_score,
            "root_cause_score": self.root_cause_score,
            "relevant_files_score": self.relevant_files_score,
            "reasoning_score": self.reasoning_score,
            "format_score": self.format_score,
            "deterministic_score": self.deterministic_score,
            "semantic_score": self.semantic_score,
            "metadata": self.metadata}


@dataclass
class InteractionResult:
    """
    Result of Turn-3 adversarial verification.

    Red and Blue receive separate interaction rewards.
    """

    red_reward: float
    blue_reward: float
    case_id: int
    red_claim_type: str
    ground_truth_state: str
    attack_claim: Optional[str] = None
    verified_attack: Optional[str] = None
    remaining_attacks: List[str] = field(default_factory=list)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "red_reward": self.red_reward,
            "blue_reward": self.blue_reward,
            "case_id": self.case_id,
            "red_claim_type": self.red_claim_type,
            "ground_truth_state": self.ground_truth_state,
            "attack_claim": self.attack_claim,
            "verified_attack": self.verified_attack,
            "remaining_attacks": self.remaining_attacks,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


# ============================================================================
# ORACLE
# ============================================================================

class Oracle:
    """
    Main SOGARL Oracle.

    The Oracle has two logically separate responsibilities:

        Normal scoring
            ↓
        candidate quality

        Interaction checking
            ↓
        adversarial defense verification

    Deterministic checks are preferred whenever possible.
    Semantic judging is used for properties that cannot reliably
    be determined with deterministic rules.
    """

    def __init__(
        self,
        deterministic_checker=None,
        semantic_judge=None,
        logger=None,
    ):
        self.deterministic_checker = deterministic_checker
        self.logger = logger
        # --------------------------------------------------------------
        # Semantic Oracle
        #
        # Prefer an explicitly injected SemanticJudge.  This is the
        # normal path used by train.py/evaluate.py and avoids any hidden
        # model construction inside the Oracle.
        #
        # If no judge was injected, retain a compatibility fallback that
        # reuses the already-loaded shared Generator model.  It never
        # loads a second base model.
        # --------------------------------------------------------------
        self.semantic_judge = semantic_judge

        if (
            self.semantic_judge is None
            and config.USE_SEMANTIC_ORACLE
        ):
            try:
                from src.oracle.semantic_judge import SemanticJudge
                from src.generation.generator import Generator

                shared_model = getattr(
                    Generator,
                    "_shared_model",
                    None,
                )
                shared_tokenizer = getattr(
                    Generator,
                    "_shared_tokenizer",
                    None,
                )
                shared_device = getattr(
                    Generator,
                    "_shared_device",
                    None,
                )

                if (
                    shared_model is not None
                    and shared_tokenizer is not None
                ):
                    self.semantic_judge = SemanticJudge(
                        model=shared_model,
                        tokenizer=shared_tokenizer,
                        device=shared_device,
                        logger=logger,
                    )

                    if logger:
                        logger.info(
                            "Semantic Oracle initialized using "
                            "the shared base model."
                        )
                elif logger:
                    logger.warning(
                        "Semantic Oracle unavailable: the shared "
                        "Generator model has not been initialized."
                    )

            except Exception as exc:
                if logger:
                    logger.warning(
                        "Failed to initialize Semantic Oracle: "
                        f"{exc}"
                    )

    # ==================================================================
    # NORMAL CANDIDATE SCORING
    # ==================================================================

    def score_attack(
        self,
        response: str,
        scenario,
    ) -> OracleScore:
        """
        Score a Red Turn-1 attack candidate.

        The reward is continuous in [0, 1].
        """

        if not response or not response.strip():
            return self._empty_score(reason="empty_attack_response")

        deterministic = self._run_deterministic_check(
            response=response,
            scenario=scenario,
            candidate_type="attack")

        semantic = self._run_semantic_check(
            response=response,
            scenario=scenario,
            candidate_type="attack")

        return self._combine_scores(
            deterministic=deterministic,
            semantic=semantic,
            candidate_type="attack")

    def score_defense(
        self,
        response: str,
        scenario,
        attack_response: Optional[str] = None,
    ) -> OracleScore:
        """
        Score a Blue Turn-2 defense candidate.

        The reward is continuous in [0, 1].

        attack_response is optional because the confidence gate
        may prevent Attack_best from being supplied to Blue.
        """

        if not response or not response.strip():
            return self._empty_score(reason="empty_defense_response")

        deterministic = self._run_deterministic_check(
            response=response,
            scenario=scenario,
            candidate_type="defense",
            attack_response=attack_response)

        semantic = self._run_semantic_check(
            response=response,
            scenario=scenario,
            candidate_type="defense",
            attack_response=attack_response)

        return self._combine_scores(
            deterministic=deterministic,
            semantic=semantic,
            candidate_type="defense")

    # ==================================================================
    # TURN 3 INTERACTION
    # ==================================================================

    def evaluate_interaction(
        self,
        response: str,
        scenario,
        blue_defense: str,
    ) -> InteractionResult:
        """
        Evaluate a Red Turn-3 challenge against a Blue defense.

        The Oracle determines:

            1. What Red claims.
            2. Whether the claimed attack is valid.
            3. Whether that attack actually remains exploitable.
            4. Whether another real vulnerability remains.
            5. Which interaction case applies.

        The final interaction rewards are discrete:

            Red  ∈ {-1, 0, +1}
            Blue ∈ {-1, 0, +1}
        """

        interaction_finding = self._get_interaction_finding(
            response=response,
            scenario=scenario,
            blue_defense=blue_defense,
        )

        return self._map_interaction_reward(
            interaction_finding
        )

    # ==================================================================
    # DETERMINISTIC EVALUATION
    # ==================================================================

    def _run_deterministic_check(self,response: str,scenario,candidate_type: str,attack_response: Optional[str] = None) -> Dict[str, Any]:
        """
        Run deterministic checks.

        If deterministic checking is disabled or the checker has
        not yet been supplied, return an empty result.

        The actual low-level checks live in:

            deterministic_checks.py
        """

        if not config.USE_DETERMINISTIC_CHECKS:
            return {}

        if self.deterministic_checker is None:
            return {}

        if candidate_type == "attack":

            result = self.deterministic_checker.check_attack(
                response=response,
                scenario=scenario,
            )

        elif candidate_type == "defense":

            result = self.deterministic_checker.check_defense(
                response=response,
                scenario=scenario,
                attack_response=attack_response,
            )

        else:
            raise ValueError(
                f"Unsupported candidate type: {candidate_type}"
            )

        return self._normalize_result(
            result
        )

    # ==================================================================
    # SEMANTIC EVALUATION
    # ==================================================================

    def _run_semantic_check(
        self,
        response: str,
        scenario,
        candidate_type: str,
        attack_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run semantic evaluation.

        The semantic judge is intentionally injected rather than
        hard-coded here.

        This keeps Oracle logic separate from model-generation logic.
        """

        if not config.USE_SEMANTIC_ORACLE:
            return {}

        if self.semantic_judge is None:
            return {}

        result = self.semantic_judge(
            response=response,
            scenario=scenario,
            candidate_type=candidate_type,
            attack_response=attack_response,
            judgments=config.ORACLE_JUDGMENTS,
            temperature=config.ORACLE_TEMPERATURE,
        )

        return self._normalize_result(
            result
        )

    # ==================================================================
    # SCORE FUSION
    # ==================================================================

    def _combine_scores(
        self,
        deterministic: Dict[str, Any],
        semantic: Dict[str, Any],
        candidate_type: str,
    ) -> OracleScore:
        """
        Combine deterministic and semantic information.

        The five final components are:

            attack label
            root cause
            relevant files
            reasoning
            format

        Each component is normalized to [0, 1].

        Final reward:

            0.30 * attack_label
          + 0.20 * root_cause
          + 0.20 * relevant_files
          + 0.20 * reasoning
          + 0.10 * format
        """

        attack_label = self._get_component(
            deterministic,
            semantic,
            "attack_label_score",
        )

        root_cause = self._get_component(
            deterministic,
            semantic,
            "root_cause_score",
        )

        relevant_files = self._get_component(
            deterministic,
            semantic,
            "relevant_files_score",
        )

        reasoning = self._get_component(
            deterministic,
            semantic,
            "reasoning_score",
        )

        format_score = self._get_component(
            deterministic,
            semantic,
            "format_score",
        )

        reward = (
            config.ATTACK_LABEL_WEIGHT
            * attack_label
            +
            config.ROOT_CAUSE_WEIGHT
            * root_cause
            +
            config.RELEVANT_FILES_WEIGHT
            * relevant_files
            +
            config.REASONING_WEIGHT
            * reasoning
            +
            config.FORMAT_WEIGHT
            * format_score
        )

        reward = self._clip01(
            reward
        )

        deterministic_score = (
            self._extract_overall_score(
                deterministic
            )
        )

        semantic_score = (
            self._extract_overall_score(
                semantic
            )
        )

        metadata = {
            "candidate_type": candidate_type,
            "deterministic_available": bool(
                deterministic
            ),
            "semantic_available": bool(
                semantic
            ),
        }

        return OracleScore(
            reward=reward,
            attack_label_score=attack_label,
            root_cause_score=root_cause,
            relevant_files_score=relevant_files,
            reasoning_score=reasoning,
            format_score=format_score,
            deterministic_score=deterministic_score,
            semantic_score=semantic_score,
            metadata=metadata,
        )

    # ==================================================================
    # INTERACTION FINDING
    # ==================================================================

    def _get_interaction_finding(
        self,
        response: str,
        scenario,
        blue_defense: str,
    ) -> Dict[str, Any]:
        """
        Determine the factual state of a Turn-3 challenge.

        Expected output:

        {
            "red_claim_type": "attack" | "no_attack",
            "claimed_attack": "...",
            "claimed_attack_valid": True/False,
            "claimed_attack_remains": True/False,
            "no_attack_verified": True/False,
            "real_attack_remains": True/False,
            "remaining_attacks": [...],
            "verified_attack": "...",
            "explanation": "..."
        }

        Interaction reward mapping happens separately.
        """

        deterministic = {}

        # --------------------------------------------------------------
        # Deterministic interaction checking is retained as a fallback.
        #
        # Semantic interaction checking is preferred because Turn 3
        # requires semantic verification of whether a vulnerability
        # remains AFTER the defense.
        # --------------------------------------------------------------
        if (
            config.USE_DETERMINISTIC_CHECKS
            and self.deterministic_checker is not None
        ):

            result = (
                self.deterministic_checker.check_interaction(
                    response=response,
                    scenario=scenario,
                    blue_defense=blue_defense,
                )
            )

            deterministic = self._normalize_result(
                result
            )

        # --------------------------------------------------------------
        # Semantic interaction verification.
        # --------------------------------------------------------------
        if (
            config.USE_SEMANTIC_ORACLE
            and self.semantic_judge is not None
        ):

            result = self.semantic_judge(
                response=response,
                scenario=scenario,
                candidate_type="interaction",
                blue_defense=blue_defense,
                judgments=config.ORACLE_JUDGMENTS,
                temperature=config.ORACLE_TEMPERATURE,
            )

            semantic = self._normalize_result(
                result
            )

            required_interaction_fields = {
                "red_claim_type",
                "real_attack_remains",
            }

            if semantic and required_interaction_fields.issubset(
                semantic.keys()
            ):
                return semantic

            if logger := self.logger:
                logger.warning(
                    "Semantic interaction result was empty or "
                    "incomplete; falling back to deterministic "
                    "interaction checking."
                )

        # --------------------------------------------------------------
        # Fall back to deterministic interaction evaluation only when
        # semantic evaluation is unavailable or returned no result.
        # --------------------------------------------------------------
        if deterministic:
            return deterministic

        raise RuntimeError(
            "Oracle could not evaluate Turn-3 interaction. "
            "A deterministic checker or semantic judge is required."
        )

    # ==================================================================
    # INTERACTION REWARD MAPPING
    # ==================================================================

    def _map_interaction_reward(
        self,
        finding: Dict[str, Any],
    ) -> InteractionResult:
        """
        Convert factual interaction findings into the locked
        five-case reward matrix.

        Locked V1 matrix:

        Case 1:
            Correct attack remains
            Red  +1
            Blue -1

        Case 2:
            Fabricated attack
            No attack remains
            Red  -1
            Blue +1

        Case 3:
            Wrong attack named
            Different real attack remains
            Red   0
            Blue -1

        Case 4:
            No attack claimed
            No attack remains
            Red  +1
            Blue +1

        Case 5:
            No attack claimed
            Real vulnerability remains
            Red  -1
            Blue -1
        """

        claim_type = self._normalize_claim_type(
            finding
        )

        claimed_attack = finding.get(
            "claimed_attack"
        )

        verified_attack = finding.get(
            "verified_attack"
        )

        remaining_attacks = finding.get(
            "remaining_attacks",
            []
        )

        if remaining_attacks is None:
            remaining_attacks = []

        remaining_attacks = list(
            remaining_attacks
        )

        # Preserve tri-state interaction state: True/False/None.
        # Never convert an indeterminate None into False.
        real_attack_remains = finding.get(
            "real_attack_remains",
            None,
        )

        if real_attack_remains is None:
            real_attack_remains = bool(remaining_attacks)
        elif not isinstance(real_attack_remains, bool):
            real_attack_remains = bool(real_attack_remains)

        claimed_attack_valid = bool(
            finding.get(
                "claimed_attack_valid",
                False,
            )
        )

        claimed_attack_remains = bool(
            finding.get(
                "claimed_attack_remains",
                False,
            )
        )

        no_attack_verified = finding.get(
            "no_attack_verified",
            None,
        )

        if no_attack_verified is None:
            no_attack_verified = not real_attack_remains
        elif not isinstance(no_attack_verified, bool):
            no_attack_verified = bool(no_attack_verified)

        # --------------------------------------------------------------
        # CASE 1
        # --------------------------------------------------------------

        if (
            claim_type == "attack"
            and claimed_attack_valid
        ):
            return InteractionResult(
                red_reward=(
                    config.INTERACTION_REWARD_CORRECT_ATTACK
                ),
                blue_reward=(
                    config.BLUE_REWARD_VALID_ATTACK_REMAINS
                ),
                case_id=1,
                red_claim_type="attack",
                ground_truth_state="valid_attack_remains",
                attack_claim=claimed_attack,
                verified_attack=verified_attack,
                remaining_attacks=remaining_attacks,
                explanation=(
                    "Red correctly identified a valid attack "
                    "that remains exploitable."
                ),
            )

        # --------------------------------------------------------------
        # CASE 2
        # --------------------------------------------------------------

        if (
            claim_type == "attack"
            and not claimed_attack_valid
            and not real_attack_remains
        ):
            return InteractionResult(
                red_reward=(
                    config.INTERACTION_REWARD_WRONG_ATTACK_NO_REMAINING
                ),
                blue_reward=(
                    config.BLUE_REWARD_NO_ATTACK_REMAINS
                ),
                case_id=2,
                red_claim_type="attack",
                ground_truth_state="no_attack_remains",
                attack_claim=claimed_attack,
                verified_attack=verified_attack,
                remaining_attacks=remaining_attacks,
                explanation=(
                    "Red fabricated an attack, but the Blue "
                    "defense genuinely closes the vulnerability."
                ),
            )

        # --------------------------------------------------------------
        # CASE 3
        # --------------------------------------------------------------

        if (
            claim_type == "attack"
            and not claimed_attack_valid
            and real_attack_remains
        ):
            return InteractionResult(
                red_reward=(
                    config.INTERACTION_REWARD_WRONG_ATTACK_REAL_REMAINING
                ),
                blue_reward=-1.0,
                case_id=3,
                red_claim_type="attack",
                ground_truth_state="different_attack_remains",
                attack_claim=claimed_attack,
                verified_attack=verified_attack,
                remaining_attacks=remaining_attacks,
                explanation=(
                    "Red identified that the defense is not "
                    "fully safe but named the wrong attack. "
                    "The defense remains objectively broken."
                ),
            )

        # --------------------------------------------------------------
        # CASE 4
        # --------------------------------------------------------------

        if (
            claim_type == "no_attack"
            and no_attack_verified
            and not real_attack_remains
        ):
            return InteractionResult(
                red_reward=(
                    config.INTERACTION_REWARD_NO_ATTACK_CORRECT
                ),
                blue_reward=(
                    config.BLUE_REWARD_NO_ATTACK_REMAINS
                ),
                case_id=4,
                red_claim_type="no_attack",
                ground_truth_state="no_attack_remains",
                attack_claim=None,
                verified_attack=verified_attack,
                remaining_attacks=remaining_attacks,
                explanation=(
                    "Red correctly verified that the defense "
                    "closes the vulnerability."
                ),
            )

        # --------------------------------------------------------------
        # CASE 5
        # --------------------------------------------------------------

        if (
            claim_type == "no_attack"
            and real_attack_remains
        ):
            return InteractionResult(
                red_reward=(
                    config.INTERACTION_REWARD_NO_ATTACK_MISSED
                ),
                blue_reward=-1.0,
                case_id=5,
                red_claim_type="no_attack",
                ground_truth_state="real_attack_missed",
                attack_claim=None,
                verified_attack=verified_attack,
                remaining_attacks=remaining_attacks,
                explanation=(
                    "Red incorrectly concluded that no attack "
                    "remains, while a real vulnerability remains."
                ),
            )

        raise ValueError(
            "Interaction finding does not match any "
            "defined SOGARL interaction case."
        )

    # ==================================================================
    # HELPERS
    # ==================================================================

    @staticmethod
    def _normalize_claim_type(
        finding: Dict[str, Any],
    ) -> str:
        """
        Normalize the Red Turn-3 claim type.
        """

        claim_type = finding.get(
            "red_claim_type",
            finding.get(
                "claim_type",
                "",
            ),
        )

        claim_type = str(
            claim_type
        ).strip().lower()

        if claim_type in {
            "none",
            "no_attack",
            "no attack",
            "no vulnerability",
            "no_vulnerability",
        }:
            return "no_attack"

        if claim_type in {
            "attack",
            "attack_exists",
            "attack exists",
            "vulnerability",
            "vulnerability_exists",
        }:
            return "attack"

        raise ValueError(
            f"Unknown interaction claim type: {claim_type}"
        )

    @staticmethod
    def _normalize_result(
        result: Any,
    ) -> Dict[str, Any]:
        """
        Convert checker results into a dictionary.
        """

        if result is None:
            return {}

        if isinstance(
            result,
            dict,
        ):

            return result

        if hasattr(
            result,
            "to_dict",
        ):

            return result.to_dict()

        if hasattr(
            result,
            "__dict__",
        ):

            return dict(
                result.__dict__
            )

        raise TypeError(
            "Oracle checker result must be a dictionary "
            "or expose to_dict()/__dict__."
        )

    @staticmethod
    def _get_component(
        deterministic: Dict[str, Any],
        semantic: Dict[str, Any],
        key: str,
    ) -> float:
        """
        Obtain a component score.

        Deterministic result takes priority.

        Semantic result fills the gap when deterministic
        evaluation is unavailable or explicitly returns None.
        """

        if (
            key in deterministic
            and deterministic[key] is not None
        ):

            return Oracle._clip01(
                deterministic[key]
            )

        if (
            key in semantic
            and semantic[key] is not None
        ):

            return Oracle._clip01(
                semantic[key]
            )

        return 0.0

    @staticmethod
    def _extract_overall_score(
        result: Dict[str, Any],
    ) -> Optional[float]:
        """
        Extract optional overall checker score.
        """

        if not result:
            return None

        for key in (
            "overall_score",
            "score",
            "reward",
        ):

            if (
                key in result
                and result[key] is not None
            ):

                return Oracle._clip01(
                    result[key]
                )

        return None

    @staticmethod
    def _clip01(
        value: Any,
    ) -> float:
        """
        Clamp a value to [0, 1].
        """

        try:

            value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    @staticmethod
    def _empty_score(
        reason: str,
    ) -> OracleScore:
        """
        Return a zero reward for an invalid response.
        """

        return OracleScore(
            reward=0.0,
            metadata={
                "reason": reason
            },
        )