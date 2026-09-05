"""
interaction_checker.py

Maps verified Turn-3 interaction findings to the SOGARL
five-case interaction reward matrix.

Responsibilities
----------------
This module:

    - receives factual Turn-3 findings
    - validates the finding
    - determines Case 1–5
    - assigns Red interaction reward
    - assigns Blue interaction reward
    - returns a structured InteractionResult

This module does NOT:

    - inspect source code
    - load metadata
    - call an LLM
    - generate Red responses
    - generate Blue responses
    - determine vulnerability ground truth
    - calculate GRPO advantages
    - normalize rewards
    - update model parameters

The factual security evaluation belongs to:

    deterministic_checks.py
        +
    semantic Oracle

This module only maps the verified factual state to the
locked interaction reward matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from configs import config


# ============================================================================
# INTERACTION RESULT
# ============================================================================

@dataclass
class InteractionOutcome:
    """
    Final result of one Turn-3 Red challenge against one Blue defense.

    Reward values are the raw interaction rewards:

        -1
         0
        +1

    These values are NOT normalized here.

    Reward normalization and advantage calculation belong to
    RewardManager / GRPO components.
    """

    case_id: int

    red_reward: float

    blue_reward: float

    red_claim_type: str

    ground_truth_state: str

    claimed_attack: Optional[str] = None

    verified_attack: Optional[str] = None

    remaining_attacks: List[str] = field(
        default_factory=list
    )

    explanation: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the interaction outcome to a dictionary."""

        return {
            "case_id": self.case_id,
            "red_reward": self.red_reward,
            "blue_reward": self.blue_reward,
            "red_claim_type": self.red_claim_type,
            "ground_truth_state": self.ground_truth_state,
            "claimed_attack": self.claimed_attack,
            "verified_attack": self.verified_attack,
            "remaining_attacks": self.remaining_attacks,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


# ============================================================================
# INTERACTION CHECKER
# ============================================================================

class InteractionChecker:
    """
    Apply the locked SOGARL Turn-3 interaction matrix.

    Five possible cases exist.

    ----------------------------------------------------------------------
    CASE 1
    ----------------------------------------------------------------------

    Red correctly identifies a valid attack that remains.

        Red  = +1
        Blue = -1


    ----------------------------------------------------------------------
    CASE 2
    ----------------------------------------------------------------------

    Red claims an attack, but the claimed attack does not remain
    after the Blue defense and no real attack remains.

        Red  = -1
        Blue = +1


    ----------------------------------------------------------------------
    CASE 3
    ----------------------------------------------------------------------

    Red identifies the wrong attack, but another real attack remains.

        Red  = 0
        Blue = -1


    ----------------------------------------------------------------------
    CASE 4
    ----------------------------------------------------------------------

    Red correctly says that no attack remains.

        Red  = +1
        Blue = +1


    ----------------------------------------------------------------------
    CASE 5
    ----------------------------------------------------------------------

    Red says that no attack remains, but a real attack remains.

        Red  = -1
        Blue = -1


    IMPORTANT
    ---------

    This class does not decide whether an attack actually remains.

    That factual information must already be present in the
    supplied finding.
    """

    def __init__(self, logger=None):
        self.logger = logger

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def check(
        self,
        finding: Dict[str, Any],
    ) -> InteractionOutcome:
        """
        Convert a verified Turn-3 finding into an interaction outcome.

        Expected finding structure:

        {
            "red_claim_type": "attack" | "no_attack",

            "claimed_attack": "...",

            "claimed_attack_valid": True | False,

            "claimed_attack_remains": True | False,

            "no_attack_verified": True | False,

            "real_attack_remains": True | False,

            "remaining_attacks": ["..."],

            "verified_attack": "..."
        }

        The checker then determines Case 1–5.
        """

        finding = self._validate_finding(
            finding
        )

        claim_type = self._normalize_claim_type(
            finding.get(
                "red_claim_type",
                finding.get(
                    "claim_type"
                ),
            )
        )

        claimed_attack = finding.get(
            "claimed_attack"
        )

        verified_attack = finding.get(
            "verified_attack"
        )

        remaining_attacks_value = finding.get(
            "remaining_attacks",
            [],
        )

        if remaining_attacks_value is None:

            remaining_attacks = []

        elif isinstance(
            remaining_attacks_value,
            list,
        ):

            if not all(
                isinstance(
                    attack,
                    str,
                )
                for attack
                in remaining_attacks_value
            ):

                raise TypeError(
                    "'remaining_attacks' must contain "
                    "only strings."
                )

            remaining_attacks = list(
                remaining_attacks_value
            )

        else:

            raise TypeError(
                "'remaining_attacks' must be a list of strings."
            )

        claimed_attack_valid = (
            self._get_bool_field(
                finding,
                "claimed_attack_valid",
                default=False,
            )
        )

        claimed_attack_remains = (
            self._get_bool_field(
                finding,
                "claimed_attack_remains",
                default=False,
            )
        )

        if "real_attack_remains" in finding:

            real_attack_remains = (
                self._get_bool_field(
                    finding,
                    "real_attack_remains",
                    default=False,
                )
            )

        else:

            real_attack_remains = bool(
                remaining_attacks
            )

        if "no_attack_verified" in finding:

            no_attack_verified = (
                self._get_bool_field(
                    finding,
                    "no_attack_verified",
                    default=False,
                )
            )

        else:

            no_attack_verified = (
                not real_attack_remains
            )

        # ====================================================================
        # CASE 1
        # ====================================================================

        if (
            claim_type == "attack"
            and claimed_attack_valid
            and claimed_attack_remains
            and real_attack_remains
        ):

            outcome = self._case_1(
                claimed_attack=claimed_attack,
                verified_attack=verified_attack,
                remaining_attacks=remaining_attacks,
            )

            self._log_outcome(
                outcome
            )

            return outcome

        # ====================================================================
        # CASE 2
        # ====================================================================

        if (
            claim_type == "attack"
            and not claimed_attack_remains
            and not real_attack_remains
        ):

            outcome = self._case_2(
                claimed_attack=claimed_attack,
                verified_attack=verified_attack,
                remaining_attacks=remaining_attacks,
            )

            self._log_outcome(
                outcome
            )

            return outcome

        # ====================================================================
        # CASE 3
        # ====================================================================

        if (
            claim_type == "attack"
            and not claimed_attack_remains
            and real_attack_remains
        ):

            outcome = self._case_3(
                claimed_attack=claimed_attack,
                verified_attack=verified_attack,
                remaining_attacks=remaining_attacks,
            )

            self._log_outcome(
                outcome
            )

            return outcome

        # ====================================================================
        # CASE 4
        # ====================================================================

        if (
            claim_type == "no_attack"
            and no_attack_verified
            and not real_attack_remains
        ):

            outcome = self._case_4(
                verified_attack=verified_attack,
                remaining_attacks=remaining_attacks,
            )

            self._log_outcome(
                outcome
            )

            return outcome

        # ====================================================================
        # CASE 5
        # ====================================================================

        if (
            claim_type == "no_attack"
            and real_attack_remains
        ):

            outcome = self._case_5(
                verified_attack=verified_attack,
                remaining_attacks=remaining_attacks,
            )

            self._log_outcome(
                outcome
            )

            return outcome

        # ====================================================================
        # INVALID / INCONSISTENT FINDING
        # ====================================================================

        raise ValueError(
            "Turn-3 interaction finding does not match "
            "any SOGARL interaction case.\n"
            f"Finding: {finding}"
        )

    # ========================================================================
    # CASE 1
    # ========================================================================

    def _case_1(
        self,
        claimed_attack: Optional[str],
        verified_attack: Optional[str],
        remaining_attacks: List[str],
    ) -> InteractionOutcome:
        """
        Case 1:

        Red correctly identifies a valid attack that remains.

            Red  = +1
            Blue = -1
        """

        return InteractionOutcome(
            case_id=1,
            red_reward=(
                config.INTERACTION_REWARD_CORRECT_ATTACK
            ),
            blue_reward=(
                config.BLUE_REWARD_VALID_ATTACK_REMAINS
            ),
            red_claim_type="attack",
            ground_truth_state=(
                "valid_attack_remains"
            ),
            claimed_attack=claimed_attack,
            verified_attack=verified_attack,
            remaining_attacks=remaining_attacks,
            explanation=(
                "Red correctly identified a valid attack "
                "that remains exploitable after the Blue defense."
            ),
            metadata={
                "outcome": "correct_attack_remains",
                "red_correct": True,
                "blue_defense_failed": True,
            },
        )

    # ========================================================================
    # CASE 2
    # ========================================================================

    def _case_2(
        self,
        claimed_attack: Optional[str],
        verified_attack: Optional[str],
        remaining_attacks: List[str],
    ) -> InteractionOutcome:
        """
        Case 2:

        Red claims an attack, but the claimed attack does not remain
        after the Blue defense and no real attack remains.

            Red  = -1
            Blue = +1

        This includes both:

            - a fabricated/invalid attack when nothing real remains
            - a previously valid attack that the Blue defense
              successfully fixed
        """

        return InteractionOutcome(
            case_id=2,
            red_reward=(
                config.INTERACTION_REWARD_WRONG_ATTACK_NO_REMAINING
            ),
            blue_reward=(
                config.BLUE_REWARD_NO_ATTACK_REMAINS
            ),
            red_claim_type="attack",
            ground_truth_state=(
                "no_attack_remains"
            ),
            claimed_attack=claimed_attack,
            verified_attack=verified_attack,
            remaining_attacks=remaining_attacks,
            explanation=(
                "Red claimed an attack that does not remain "
                "after the Blue defense, and no real "
                "vulnerability remains."
            ),
            metadata={
                "outcome": (
                    "claimed_attack_not_remaining"
                ),
                "red_correct": False,
                "blue_defense_success": True,
            },
        )

    # ========================================================================
    # CASE 3
    # ========================================================================

    def _case_3(
        self,
        claimed_attack: Optional[str],
        verified_attack: Optional[str],
        remaining_attacks: List[str],
    ) -> InteractionOutcome:
        """
        Case 3:

        Red identifies the wrong attack, but a different
        real attack remains.

            Red  = 0
            Blue = -1

        Red gets neutral reward because it failed to identify
        the actual remaining vulnerability.

        Blue gets negative reward because the defense is
        objectively incomplete.
        """

        return InteractionOutcome(
            case_id=3,
            red_reward=(
                config.INTERACTION_REWARD_WRONG_ATTACK_REAL_REMAINING
            ),
            blue_reward=(
                config.BLUE_REWARD_VALID_ATTACK_REMAINS
            ),
            red_claim_type="attack",
            ground_truth_state=(
                "different_attack_remains"
            ),
            claimed_attack=claimed_attack,
            verified_attack=verified_attack,
            remaining_attacks=remaining_attacks,
            explanation=(
                "Red identified an incorrect attack, but a "
                "different real vulnerability remains. "
                "The Blue defense is objectively incomplete."
            ),
            metadata={
                "outcome": (
                    "wrong_attack_different_attack_remains"
                ),
                "red_partial_detection": True,
                "red_exact_detection": False,
                "blue_defense_failed": True,
            },
        )

    # ========================================================================
    # CASE 4
    # ========================================================================

    def _case_4(
        self,
        verified_attack: Optional[str],
        remaining_attacks: List[str],
    ) -> InteractionOutcome:
        """
        Case 4:

        Red correctly concludes that no attack remains.

            Red  = +1
            Blue = +1
        """

        return InteractionOutcome(
            case_id=4,
            red_reward=(
                config.INTERACTION_REWARD_NO_ATTACK_CORRECT
            ),
            blue_reward=(
                config.BLUE_REWARD_NO_ATTACK_REMAINS
            ),
            red_claim_type="no_attack",
            ground_truth_state=(
                "no_attack_remains"
            ),
            claimed_attack=None,
            verified_attack=verified_attack,
            remaining_attacks=remaining_attacks,
            explanation=(
                "Red correctly verified that the Blue defense "
                "eliminates the exploitable vulnerability."
            ),
            metadata={
                "outcome": "defense_verified",
                "red_correct": True,
                "blue_defense_success": True,
            },
        )

    # ========================================================================
    # CASE 5
    # ========================================================================

    def _case_5(
        self,
        verified_attack: Optional[str],
        remaining_attacks: List[str],
    ) -> InteractionOutcome:
        """
        Case 5:

        Red incorrectly concludes that no attack remains,
        while a real vulnerability remains.

            Red  = -1
            Blue = -1
        """

        return InteractionOutcome(
            case_id=5,
            red_reward=(
                config.INTERACTION_REWARD_NO_ATTACK_MISSED
            ),
            blue_reward=(
                config.BLUE_REWARD_VALID_ATTACK_REMAINS
            ),
            red_claim_type="no_attack",
            ground_truth_state=(
                "real_attack_missed"
            ),
            claimed_attack=None,
            verified_attack=verified_attack,
            remaining_attacks=remaining_attacks,
            explanation=(
                "Red incorrectly concluded that no attack "
                "remains while a real vulnerability remains."
            ),
            metadata={
                "outcome": (
                    "missed_remaining_vulnerability"
                ),
                "red_detection_failed": True,
                "blue_defense_failed": True,
            },
        )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_finding(
        finding: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate the basic structure of a factual finding.

        This method intentionally performs structural validation
        only.

        It does NOT determine whether the finding is factually
        correct.
        """

        if not isinstance(
            finding,
            dict,
        ):

            raise TypeError(
                "Interaction finding must be a dictionary."
            )

        if (
            "red_claim_type" not in finding
            and "claim_type" not in finding
        ):

            raise ValueError(
                "Interaction finding must contain "
                "'red_claim_type'."
            )

        required_fields = {
            "claimed_attack_valid",
            "claimed_attack_remains",
            "no_attack_verified",
            "real_attack_remains",
            "remaining_attacks",
        }

        missing = (
            required_fields
            - set(
                finding.keys()
            )
        )

        if missing:

            raise ValueError(
                "Interaction finding is missing required fields: "
                f"{sorted(missing)}"
            )

        # If both aliases are supplied, they must agree.
        if (
            "red_claim_type" in finding
            and "claim_type" in finding
        ):

            red_claim_type = (
                InteractionChecker
                ._normalize_claim_type(
                    finding.get(
                        "red_claim_type"
                    )
                )
            )

            claim_type = (
                InteractionChecker
                ._normalize_claim_type(
                    finding.get(
                        "claim_type"
                    )
                )
            )

            if red_claim_type != claim_type:

                raise ValueError(
                    "'red_claim_type' and 'claim_type' "
                    "describe different claim types."
                )

        return finding

    # ========================================================================
    # STRICT BOOLEAN VALIDATION
    # ========================================================================

    @staticmethod
    def _get_bool_field(
        finding: Dict[str, Any],
        field_name: str,
        default: bool,
    ) -> bool:
        """
        Read a factual boolean without coercing arbitrary values.

        In particular, bool("false") is True in Python, so string
        representations of booleans must not silently alter the
        interaction classification.
        """

        value = finding.get(
            field_name,
            default,
        )

        if not isinstance(
            value,
            bool,
        ):

            raise TypeError(
                f"'{field_name}' must be a boolean, "
                f"got {type(value).__name__}."
            )

        return value

    # ========================================================================
    # CLAIM TYPE NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_claim_type(
        claim_type: Any,
    ) -> str:
        """
        Normalize Turn-3 Red claim types.

        Accepted attack forms:

            attack
            attack_exists
            vulnerability
            vulnerability_exists

        Accepted no-attack forms:

            no_attack
            no attack
            none
            no_vulnerability
            no vulnerability
        """

        if claim_type is None:

            raise ValueError(
                "red_claim_type cannot be None."
            )

        value = str(
            claim_type
        ).strip().lower()

        if value in {
            "attack",
            "attack_exists",
            "attack exists",
            "vulnerability",
            "vulnerability_exists",
            "vulnerability exists",
        }:

            return "attack"

        if value in {
            "no_attack",
            "no attack",
            "none",
            "no_vulnerability",
            "no vulnerability",
            "no vulnerability remains",
            "no attack remains",
        }:

            return "no_attack"

        raise ValueError(
            f"Unknown interaction claim type: "
            f"{claim_type}"
        )

    # ========================================================================
    # LOGGING
    # ========================================================================

    def _log_outcome(
        self,
        outcome: InteractionOutcome,
    ) -> None:
        """
        Log the final interaction classification.

        The full response/finding is deliberately not logged here.
        """

        if self.logger is None:

            return

        self.logger.info(
            "Interaction Case "
            f"{outcome.case_id}: "
            f"Red={outcome.red_reward:+.1f}, "
            f"Blue={outcome.blue_reward:+.1f}"
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def check_interaction(finding: Dict[str, Any],logger=None) -> InteractionOutcome:
    """
    Convenience wrapper around InteractionChecker.

    Example
    -------

        outcome = check_interaction(
            finding
        )

        print(outcome.case_id)
        print(outcome.red_reward)
        print(outcome.blue_reward)
    """

    checker = InteractionChecker(logger=logger)
    return checker.check(finding)