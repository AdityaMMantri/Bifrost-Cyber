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
from src.oracle.interaction_checker import InteractionChecker

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
            "metadata": self.metadata,
        }


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

    def __init__(self,deterministic_checker=None,semantic_judge=None,logger=None):
        self.deterministic_checker = (deterministic_checker)
        self.logger = logger
        self.semantic_judge = (semantic_judge)

        # --------------------------------------------------------------
        # Turn-3 interaction mapper.
        #
        # InteractionChecker is the single authority for mapping a
        # verified factual Turn-3 state into Case 1-5 rewards.
        #
        # This does not change Oracle's public API. Oracle continues
        # returning InteractionResult.
        # --------------------------------------------------------------
        self.interaction_checker = (InteractionChecker(logger=logger))
        # --------------------------------------------------------------
        # Semantic Oracle
        #
        # Prefer an explicitly injected SemanticJudge.
        #
        # If no judge was injected, retain a compatibility fallback that
        # reuses the already-loaded shared Generator model. It never
        # loads a second base model.
        # --------------------------------------------------------------

        if (
            self.semantic_judge is None
            and config.USE_SEMANTIC_ORACLE
        ):

            try:

                from src.oracle.semantic_judge import (
                    SemanticJudge
                )

                from src.generation.generator import (
                    Generator
                )

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
                    and shared_device is not None
                ):

                    self.semantic_judge = (
                        SemanticJudge(
                            model=shared_model,
                            tokenizer=shared_tokenizer,
                            device=shared_device,
                            logger=logger,
                        )
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

        if (
            not response
            or not response.strip()
        ):

            return self._empty_score(
                reason="empty_attack_response"
            )

        deterministic = (
            self._run_deterministic_check(
                response=response,
                scenario=scenario,
                candidate_type="attack",
            )
        )

        semantic = (
            self._run_semantic_check(
                response=response,
                scenario=scenario,
                candidate_type="attack",
            )
        )

        return self._combine_scores(
            deterministic=deterministic,
            semantic=semantic,
            candidate_type="attack",
        )

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

        if (
            not response
            or not response.strip()
        ):

            return self._empty_score(
                reason="empty_defense_response"
            )

        deterministic = (
            self._run_deterministic_check(
                response=response,
                scenario=scenario,
                candidate_type="defense",
                attack_response=attack_response,
            )
        )

        semantic = (
            self._run_semantic_check(
                response=response,
                scenario=scenario,
                candidate_type="defense",
                attack_response=attack_response,
            )
        )

        return self._combine_scores(
            deterministic=deterministic,
            semantic=semantic,
            candidate_type="defense",
        )

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

        interaction_finding = (
            self._get_interaction_finding(
                response=response,
                scenario=scenario,
                blue_defense=blue_defense,
            )
        )

        return self._map_interaction_reward(
            interaction_finding
        )

    # ==================================================================
    # DETERMINISTIC EVALUATION
    # ==================================================================

    def _run_deterministic_check(
        self,
        response: str,
        scenario,
        candidate_type: str,
        attack_response: Optional[str] = None,
    ) -> Dict[str, Any]:
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

            result = (
                self.deterministic_checker
                .check_attack(
                    response=response,
                    scenario=scenario,
                )
            )

        elif candidate_type == "defense":

            result = (
                self.deterministic_checker
                .check_defense(
                    response=response,
                    scenario=scenario,
                    attack_response=attack_response,
                )
            )

        else:

            raise ValueError(
                f"Unsupported candidate type: "
                f"{candidate_type}"
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

        # --------------------------------------------------------------
        # A SemanticJudge may return a non-empty dictionary while
        # explicitly marking semantic evaluation unavailable.
        #
        # Do not use bool(semantic) alone because:
        #
        # {
        #     "semantic_score": None,
        #     "semantic_available": False,
        # }
        #
        # is truthy as a dictionary.
        # --------------------------------------------------------------

        semantic_available = bool(
            semantic.get(
                "semantic_available",
                bool(semantic),
            )
        )

        metadata = {
            "candidate_type": candidate_type,
            "deterministic_available": bool(
                deterministic
            ),
            "semantic_available": (
                semantic_available
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
        # Deterministic interaction checking.
        # --------------------------------------------------------------

        if (
            config.USE_DETERMINISTIC_CHECKS
            and self.deterministic_checker is not None
        ):

            result = (
                self.deterministic_checker
                .check_interaction(
                    response=response,
                    scenario=scenario,
                    blue_defense=blue_defense,
                )
            )

            deterministic = (
                self._normalize_result(
                    result
                )
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

            # ----------------------------------------------------------
            # These fields are all required by InteractionChecker.
            #
            # Accepting a partial semantic result could otherwise
            # silently manufacture False values and map to the wrong
            # reward case.
            # ----------------------------------------------------------

            required_interaction_fields = {
                "red_claim_type",
                "claimed_attack_valid",
                "claimed_attack_remains",
                "no_attack_verified",
                "real_attack_remains",
                "remaining_attacks",
            }

            semantic_complete = (
                bool(semantic)
                and required_interaction_fields
                .issubset(
                    semantic.keys()
                )
                and isinstance(
                    semantic.get(
                        "claimed_attack_valid"
                    ),
                    bool,
                )
                and isinstance(
                    semantic.get(
                        "claimed_attack_remains"
                    ),
                    bool,
                )
                and isinstance(
                    semantic.get(
                        "no_attack_verified"
                    ),
                    bool,
                )
                and isinstance(
                    semantic.get(
                        "real_attack_remains"
                    ),
                    bool,
                )
                and isinstance(
                    semantic.get(
                        "remaining_attacks"
                    ),
                    list,
                )
            )

            if semantic_complete:

                return semantic

            if self.logger:

                self.logger.warning(
                    "Semantic interaction result was empty, "
                    "incomplete, or malformed; falling back "
                    "to deterministic interaction checking."
                )

        # --------------------------------------------------------------
        # Deterministic fallback.
        #
        # DeterministicChecker may intentionally return
        # real_attack_remains=None when post-defense state cannot be
        # established from metadata. Such a result must NOT be mapped
        # into a reward as though None were False.
        # --------------------------------------------------------------

        if deterministic:

            required_interaction_fields = {
                "red_claim_type",
                "claimed_attack_valid",
                "claimed_attack_remains",
                "no_attack_verified",
                "real_attack_remains",
                "remaining_attacks",
            }

            deterministic_complete = (
                required_interaction_fields
                .issubset(
                    deterministic.keys()
                )
                and isinstance(
                    deterministic.get(
                        "claimed_attack_valid"
                    ),
                    bool,
                )
                and isinstance(
                    deterministic.get(
                        "claimed_attack_remains"
                    ),
                    bool,
                )
                and isinstance(
                    deterministic.get(
                        "no_attack_verified"
                    ),
                    bool,
                )
                and isinstance(
                    deterministic.get(
                        "real_attack_remains"
                    ),
                    bool,
                )
                and isinstance(
                    deterministic.get(
                        "remaining_attacks"
                    ),
                    list,
                )
            )

            if deterministic_complete:

                return deterministic

        raise RuntimeError(
            "Oracle could not establish a complete "
            "Turn-3 interaction finding. "
            "Semantic evaluation was unavailable or incomplete, "
            "and deterministic evidence was insufficient."
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

        InteractionChecker is the single authoritative mapper.

        This method is intentionally retained so the existing
        Oracle API and internal call structure remain unchanged.
        """

        outcome = self.interaction_checker.check(
            finding
        )

        # --------------------------------------------------------------
        # Preserve Oracle's existing InteractionResult output contract.
        #
        # InteractionOutcome uses:
        #
        #     claimed_attack
        #
        # while Oracle's existing public result uses:
        #
        #     attack_claim
        #
        # --------------------------------------------------------------

        return InteractionResult(
            red_reward=outcome.red_reward,
            blue_reward=outcome.blue_reward,
            case_id=outcome.case_id,
            red_claim_type=outcome.red_claim_type,
            ground_truth_state=(
                outcome.ground_truth_state
            ),
            attack_claim=(
                outcome.claimed_attack
            ),
            verified_attack=(
                outcome.verified_attack
            ),
            remaining_attacks=list(
                outcome.remaining_attacks
            ),
            explanation=outcome.explanation,
            metadata=dict(
                outcome.metadata
            ),
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

        Retained for compatibility with existing internal/tests.
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
            f"Unknown interaction claim type: "
            f"{claim_type}"
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

        # --------------------------------------------------------------
        # Existing output contract is preserved:
        #
        # OracleScore component fields remain floats.
        #
        # Missing evidence therefore retains the previous 0.0 fallback.
        # Higher-level availability is preserved separately in metadata.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # SemanticJudge explicitly returns "semantic_score".
        #
        # deterministic_checks.py returns "overall_score".
        # --------------------------------------------------------------

        for key in (
            "semantic_score",
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

        if value != value:

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