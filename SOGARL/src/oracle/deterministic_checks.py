"""
deterministic_checks.py

Deterministic checks used by the SOGARL Oracle.

Responsibilities:
    - Extract structured claims from model responses.
    - Compare attack/defense claims against hidden scenario metadata.
    - Check relevant-file claims.
    - Check required response structure.
    - Provide deterministic evidence to oracle.py.
    - Evaluate factual Turn-3 interaction states when possible.

This module does NOT:
    - call an LLM
    - generate responses
    - perform GRPO
    - calculate advantages
    - select candidates
    - modify model parameters
    - execute the target repository
    - inspect programming-language-specific syntax

The semantic quality of reasoning is handled by the semantic
Oracle layer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set


class DeterministicChecker:
    """
    Deterministic evaluator for SOGARL model responses.

    The checker relies on the hidden Scenario metadata.

    Typical metadata:

        valid_attacks
        valid_defenses
        optimal_attack
        optimal_defense
        relevant_files
        vulnerabilities

    The checker is intentionally conservative.

    If something cannot be established reliably using deterministic
    information, it returns an incomplete result and lets the
    semantic Oracle handle the remaining reasoning.
    """

    # ==================================================================
    # PUBLIC ATTACK CHECK
    # ==================================================================

    def check_attack(
        self,
        response: str,
        scenario,
    ) -> Dict[str, Any]:
        """
        Deterministically evaluate a Red Turn-1 attack response.

        Returns component scores in [0, 1].

        Components:

            attack_label_score
            root_cause_score
            relevant_files_score
            reasoning_score
            format_score
        """

        response = self._normalize_text(response)

        if not response:
            return self._empty_result(
                "empty_attack_response"
            )

        valid_attacks = self._get_valid_attacks(
            scenario
        )

        optimal_attack = self._get_optimal_attack(
            scenario
        )

        relevant_files = self._get_relevant_files(
            scenario
        )

        claimed_attack = self.extract_attack_claim(
            response
        )

        claimed_files = self.extract_files(
            response
        )

        attack_score = self._attack_label_score(
            claimed_attack=claimed_attack,
            valid_attacks=valid_attacks,
            optimal_attack=optimal_attack,
        )

        file_score = self._relevant_file_score(
            claimed_files=claimed_files,
            relevant_files=relevant_files,
        )

        format_score = self._format_score(
            response=response,
            candidate_type="attack",
        )

        root_cause_score = self._metadata_component_score(
            response=response,
            scenario=scenario,
            component="root_cause",
        )

        reasoning_score = self._metadata_component_score(
            response=response,
            scenario=scenario,
            component="reasoning",
        )

        return {
            "attack_label_score": attack_score,
            "root_cause_score": root_cause_score,
            "relevant_files_score": file_score,
            "reasoning_score": reasoning_score,
            "format_score": format_score,
            "overall_score": self._average_available(
                [
                    attack_score,
                    root_cause_score,
                    file_score,
                    reasoning_score,
                    format_score,
                ]
            ),
            "claimed_attack": claimed_attack,
            "claimed_files": claimed_files,
            "metadata": {
                "valid_attacks": valid_attacks,
                "optimal_attack": optimal_attack,
            },
        }

    # ==================================================================
    # PUBLIC DEFENSE CHECK
    # ==================================================================

    def check_defense(
        self,
        response: str,
        scenario,
        attack_response: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Deterministically evaluate a Blue Turn-2 defense response.

        The optional attack_response is useful when the confidence
        gate allowed Attack_best to be shown to Blue.
        """

        response = self._normalize_text(response)

        if not response:
            return self._empty_result(
                "empty_defense_response"
            )

        valid_defenses = self._get_valid_defenses(
            scenario
        )

        optimal_defense = self._get_optimal_defense(
            scenario
        )

        relevant_files = self._get_relevant_files(
            scenario
        )

        claimed_defense = self.extract_defense_claim(
            response
        )

        claimed_files = self.extract_files(
            response
        )

        defense_score = self._defense_label_score(
            claimed_defense=claimed_defense,
            valid_defenses=valid_defenses,
            optimal_defense=optimal_defense,
        )

        file_score = self._relevant_file_score(
            claimed_files=claimed_files,
            relevant_files=relevant_files,
        )

        format_score = self._format_score(
            response=response,
            candidate_type="defense",
        )

        root_cause_score = self._metadata_component_score(
            response=response,
            scenario=scenario,
            component="root_cause",
        )

        reasoning_score = self._metadata_component_score(
            response=response,
            scenario=scenario,
            component="reasoning",
        )

        return {
            "attack_label_score": defense_score,
            "root_cause_score": root_cause_score,
            "relevant_files_score": file_score,
            "reasoning_score": reasoning_score,
            "format_score": format_score,
            "overall_score": self._average_available(
                [
                    defense_score,
                    root_cause_score,
                    file_score,
                    reasoning_score,
                    format_score,
                ]
            ),
            "claimed_defense": claimed_defense,
            "claimed_files": claimed_files,
            "metadata": {
                "valid_defenses": valid_defenses,
                "optimal_defense": optimal_defense,
                "attack_response_supplied": (
                    attack_response is not None
                ),
            },
        }

    # ==================================================================
    # PUBLIC TURN-3 CHECK
    # ==================================================================

    def check_interaction(
        self,
        response: str,
        scenario,
        blue_defense: str,
    ) -> Dict[str, Any]:
        """
        Deterministically analyze a Red Turn-3 challenge.

        This method determines what can be established from:

            - Red's explicit claim
            - hidden scenario ground truth
            - known defense metadata

        It does not itself assign +1 / 0 / -1.

        oracle.py maps the factual result to the locked
        interaction matrix.
        """

        response = self._normalize_text(response)

        if not response:
            return {
                "red_claim_type": "no_attack",
                "claimed_attack": None,
                "claimed_attack_valid": False,
                "claimed_attack_remains": False,
                "no_attack_verified": False,
                "real_attack_remains": self._real_attack_remains(
                    scenario,
                    blue_defense,
                ),
                "remaining_attacks": self._remaining_attacks(
                    scenario,
                    blue_defense,
                ),
                "explanation": "Empty Turn-3 response.",
            }

        claim_type = self.extract_claim_type(
            response
        )

        claimed_attack = self.extract_attack_claim(
            response
        )

        remaining_attacks = self._remaining_attacks(
            scenario,
            blue_defense,
        )

        real_attack_remains = bool(
            remaining_attacks
        )

        claimed_attack_valid = self._attack_exists(
            claimed_attack,
            self._get_valid_attacks(scenario),
        )

        claimed_attack_remains = self._attack_remains(
            claimed_attack,
            remaining_attacks,
        )

        no_attack_verified = (
            not real_attack_remains
        )

        return {
            "red_claim_type": claim_type,
            "claimed_attack": claimed_attack,
            "claimed_attack_valid": claimed_attack_valid,
            "claimed_attack_remains": claimed_attack_remains,
            "no_attack_verified": no_attack_verified,
            "real_attack_remains": real_attack_remains,
            "remaining_attacks": remaining_attacks,
            "verified_attack": (
                remaining_attacks[0]
                if remaining_attacks
                else None
            ),
            "explanation": self._interaction_explanation(
                claim_type=claim_type,
                claimed_attack=claimed_attack,
                claimed_attack_valid=claimed_attack_valid,
                claimed_attack_remains=claimed_attack_remains,
                real_attack_remains=real_attack_remains,
            ),
        }

    # ==================================================================
    # CLAIM EXTRACTION
    # ==================================================================

    def extract_attack_claim(
        self,
        response: str,
    ) -> Optional[str]:
        """
        Extract an attack label from a model response.

        Supports common formats such as:

            Attack: IDOR
            Vulnerability: SQL Injection
            Attack Type: SSRF

        Exact matching against scenario metadata happens later.
        """

        patterns = [
            r"attack\s*(?:type)?\s*[:\-]\s*(.+)",
            r"vulnerability\s*(?:type)?\s*[:\-]\s*(.+)",
            r"security\s*(?:issue|vulnerability)\s*[:\-]\s*(.+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                response,
                flags=re.IGNORECASE,
            )

            if match:
                return self._clean_claim(
                    match.group(1)
                )

        return self._find_known_attack(
            response
        )

    def extract_defense_claim(
        self,
        response: str,
    ) -> Optional[str]:
        """
        Extract a defense label when explicitly provided.

        Supports:

            Defense: ...
            Mitigation: ...
            Remediation: ...
        """

        patterns = [
            r"defense\s*[:\-]\s*(.+)",
            r"defence\s*[:\-]\s*(.+)",
            r"mitigation\s*[:\-]\s*(.+)",
            r"remediation\s*[:\-]\s*(.+)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                response,
                flags=re.IGNORECASE,
            )

            if match:
                return self._clean_claim(
                    match.group(1)
                )

        return None

    def extract_claim_type(
        self,
        response: str,
    ) -> str:
        """
        Determine whether Turn-3 Red claims:

            attack
            no_attack
        """

        no_attack_patterns = [
            r"\bno attack remains\b",
            r"\bno vulnerability remains\b",
            r"\bno exploitable vulnerability\b",
            r"\bfully mitigated\b",
            r"\bthe defense closes\b",
            r"\bdefense is effective\b",
            r"\bnot exploitable\b",
        ]

        for pattern in no_attack_patterns:
            if re.search(
                pattern,
                response,
                flags=re.IGNORECASE,
            ):
                return "no_attack"

        return "attack"

    def extract_files(
        self,
        response: str,
    ) -> List[str]:
        """
        Extract likely file paths from a response.

        This is deliberately language-agnostic.

        No extension whitelist is used.
        """

        candidates = set()

        # Paths containing a directory separator.
        path_matches = re.findall(
            r"(?:[\w.\-]+[\\/])+[\w.\-]+",
            response,
        )

        candidates.update(
            self._clean_path(x)
            for x in path_matches
        )

        # Backtick-enclosed file-like values.
        inline_matches = re.findall(
            r"`([^`]+)`",
            response,
        )

        for value in inline_matches:
            value = self._clean_path(value)

            if (
                "/" in value
                or "\\" in value
            ):
                candidates.add(value)

        return sorted(
            x for x in candidates if x
        )

    # ==================================================================
    # ATTACK / DEFENSE SCORING
    # ==================================================================

    def _attack_label_score(
        self,
        claimed_attack: Optional[str],
        valid_attacks: List[str],
        optimal_attack: Optional[str],
    ) -> float:
        """Score the claimed attack against ground truth."""

        if not claimed_attack:
            return 0.0

        if optimal_attack and self._same_label(
            claimed_attack,
            optimal_attack,
        ):
            return 1.0

        for attack in valid_attacks:
            if self._same_label(
                claimed_attack,
                attack,
            ):
                return 0.75

        return 0.0

    def _defense_label_score(
        self,
        claimed_defense: Optional[str],
        valid_defenses: List[str],
        optimal_defense: Optional[str],
    ) -> float:
        """Score a claimed defense against ground truth."""

        if not claimed_defense:
            return 0.0

        if optimal_defense and self._same_label(
            claimed_defense,
            optimal_defense,
        ):
            return 1.0

        for defense in valid_defenses:
            if self._same_label(
                claimed_defense,
                defense,
            ):
                return 0.75

        return 0.0

    def _relevant_file_score(
        self,
        claimed_files: List[str],
        relevant_files: List[str],
    ) -> float:
        """
        Calculate precision/recall-style file overlap.

        If the response names no files, score = 0.
        """

        if not claimed_files:
            return 0.0

        if not relevant_files:
            return 0.0

        claimed = {
            self._normalize_path(x)
            for x in claimed_files
        }

        relevant = {
            self._normalize_path(x)
            for x in relevant_files
        }

        overlap = claimed & relevant

        if not overlap:
            return 0.0

        precision = (
            len(overlap)
            / len(claimed)
        )

        recall = (
            len(overlap)
            / len(relevant)
        )

        if precision + recall == 0:
            return 0.0

        return (
            2 * precision * recall
            / (precision + recall)
        )

    # ==================================================================
    # FORMAT
    # ==================================================================

    def _format_score(
        self,
        response: str,
        candidate_type: str,
    ) -> float:
        """
        Basic deterministic format check.

        This does not judge reasoning quality.
        """

        if not response.strip():
            return 0.0

        score = 0.0

        if len(response.split()) >= 20:
            score += 0.25

        if re.search(
            r"\b(reason|because|root cause|evidence)\b",
            response,
            flags=re.IGNORECASE,
        ):
            score += 0.25

        if re.search(
            r"\b(file|files|code|function|method|class)\b",
            response,
            flags=re.IGNORECASE,
        ):
            score += 0.25

        if candidate_type == "attack":
            if self.extract_attack_claim(
                response
            ):
                score += 0.25

        elif candidate_type == "defense":
            if self.extract_defense_claim(
                response
            ):
                score += 0.25

        return min(
            score,
            1.0,
        )

    # ==================================================================
    # METADATA COMPONENTS
    # ==================================================================

    def _metadata_component_score(
        self,
        response: str,
        scenario,
        component: str,
    ) -> Optional[float]:
        """
        Return deterministic evidence for components that may exist
        in scenario metadata.

        If no deterministic evidence exists, return None so the
        semantic Oracle can evaluate it.
        """

        metadata = getattr(
            scenario,
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            return None

        if component == "root_cause":
            root_cause = metadata.get(
                "root_cause"
            )

            if not root_cause:
                return None

            return self._text_evidence_score(
                response,
                root_cause,
            )

        if component == "reasoning":
            reasoning_points = metadata.get(
                "reasoning_points"
            )

            if not reasoning_points:
                return None

            if isinstance(
                reasoning_points,
                str,
            ):
                reasoning_points = [
                    reasoning_points
                ]

            scores = [
                self._text_evidence_score(
                    response,
                    point,
                )
                for point in reasoning_points
            ]

            return (
                sum(scores)
                / len(scores)
                if scores
                else None
            )

        return None

    # ==================================================================
    # TURN-3 GROUND TRUTH
    # ==================================================================

    def _remaining_attacks(
        self,
        scenario,
        blue_defense: str,
    ) -> List[str]:
        """
        Determine which known attacks remain after the proposed
        defense.

        Preferred source:

            vulnerability metadata containing defense mappings.

        Fallback:

            if the metadata does not explicitly define defense
            effects, return the scenario's valid attacks.

        The fallback is intentionally conservative and should
        normally be replaced by explicit scenario metadata.
        """

        vulnerabilities = getattr(
            scenario,
            "vulnerabilities",
            [],
        )

        if not vulnerabilities:
            return list(
                self._get_valid_attacks(
                    scenario
                )
            )

        remaining = []

        for vulnerability in vulnerabilities:

            if not isinstance(
                vulnerability,
                dict,
            ):
                continue

            attack = (
                vulnerability.get(
                    "attack_type"
                )
                or vulnerability.get(
                    "attack"
                )
                or vulnerability.get(
                    "id"
                )
            )

            if not attack:
                continue

            fixed_by = vulnerability.get(
                "fixed_by",
                vulnerability.get(
                    "valid_defenses",
                    [],
                ),
            )

            if isinstance(
                fixed_by,
                str,
            ):
                fixed_by = [
                    fixed_by
                ]

            defense_matches = any(
                self._same_label(
                    blue_defense,
                    defense,
                )
                for defense in fixed_by
            )

            if not defense_matches:
                remaining.append(
                    str(attack)
                )

        return remaining

    def _real_attack_remains(
        self,
        scenario,
        blue_defense: str,
    ) -> bool:
        return bool(
            self._remaining_attacks(
                scenario,
                blue_defense,
            )
        )

    def _attack_remains(
        self,
        claimed_attack: Optional[str],
        remaining_attacks: List[str],
    ) -> bool:
        if not claimed_attack:
            return False

        return any(
            self._same_label(
                claimed_attack,
                attack,
            )
            for attack in remaining_attacks
        )

    # ==================================================================
    # SCENARIO ACCESS
    # ==================================================================

    @staticmethod
    def _get_valid_attacks(
        scenario,
    ) -> List[str]:
        attacks = getattr(
            scenario,
            "valid_attacks",
            [],
        )

        return list(attacks or [])

    @staticmethod
    def _get_valid_defenses(
        scenario,
    ) -> List[str]:
        defenses = getattr(
            scenario,
            "valid_defenses",
            [],
        )

        return list(defenses or [])

    @staticmethod
    def _get_optimal_attack(
        scenario,
    ) -> Optional[str]:
        return getattr(
            scenario,
            "optimal_attack",
            None,
        )

    @staticmethod
    def _get_optimal_defense(
        scenario,
    ) -> Optional[str]:
        return getattr(
            scenario,
            "optimal_defense",
            None,
        )

    @staticmethod
    def _get_relevant_files(
        scenario,
    ) -> List[str]:
        files = getattr(
            scenario,
            "relevant_files",
            [],
        )

        return list(files or [])

    # ==================================================================
    # GENERAL HELPERS
    # ==================================================================

    @staticmethod
    def _find_known_attack(
        response: str,
    ) -> Optional[str]:
        """
        Generic fallback.

        Exact known-attack matching is preferably performed by
        _attack_label_score against scenario metadata.
        """

        match = re.search(
            r"\b(IDOR|SSRF|XSS|SQL\s+Injection|"
            r"Command\s+Injection|Path\s+Traversal|"
            r"CSRF|RCE|JWT\s+Forgery|"
            r"Broken\s+Authorization)\b",
            response,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

        return None

    @staticmethod
    def _attack_exists(
        claimed_attack: Optional[str],
        valid_attacks: List[str],
    ) -> bool:
        if not claimed_attack:
            return False

        return any(
            DeterministicChecker._same_label(
                claimed_attack,
                attack,
            )
            for attack in valid_attacks
        )

    @staticmethod
    def _same_label(
        first: Optional[str],
        second: Optional[str],
    ) -> bool:
        if not first or not second:
            return False

        return (
            DeterministicChecker._normalize_label(
                first
            )
            ==
            DeterministicChecker._normalize_label(
                second
            )
        )

    @staticmethod
    def _normalize_label(
        value: str,
    ) -> str:
        value = str(value).lower().strip()

        value = re.sub(
            r"[_\-]+",
            " ",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value

    @staticmethod
    def _normalize_path(
        value: str,
    ) -> str:
        return (
            value
            .replace("\\", "/")
            .strip()
            .lower()
        )

    @staticmethod
    def _clean_path(
        value: str,
    ) -> str:
        value = value.strip(
            " `\"'.,:;()[]{}"
        )

        return value

    @staticmethod
    def _clean_claim(
        value: str,
    ) -> str:
        value = value.strip()

        # Stop at common response-section boundaries.
        value = re.split(
            r"\n|(?=\b(?:Root Cause|"
            r"Relevant Files|Reasoning|"
            r"Evidence|Explanation)\s*:)",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        return value.strip(
            " `\"'.,:;"
        )

    @staticmethod
    def _normalize_text(
        value: Optional[str],
    ) -> str:
        if value is None:
            return ""

        return str(value).strip()

    @staticmethod
    def _text_evidence_score(
        response: str,
        expected: Any,
    ) -> float:
        """
        Simple deterministic textual evidence score.

        This is deliberately conservative.

        It is not intended to replace semantic reasoning.
        """

        if not expected:
            return 0.0

        response_normalized = (
            response.lower()
        )

        if isinstance(
            expected,
            str,
        ):
            expected_values = [
                expected
            ]
        else:
            expected_values = list(
                expected
            )

        matches = 0

        for value in expected_values:

            value = str(value).strip()

            if not value:
                continue

            if value.lower() in response_normalized:
                matches += 1

        if not expected_values:
            return 0.0

        return min(
            matches / len(expected_values),
            1.0,
        )

    @staticmethod
    def _average_available(
        values: List[Optional[float]],
    ) -> float:
        available = [
            float(value)
            for value in values
            if value is not None
        ]

        if not available:
            return 0.0

        return (
            sum(available)
            / len(available)
        )

    @staticmethod
    def _empty_result(
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "attack_label_score": 0.0,
            "root_cause_score": 0.0,
            "relevant_files_score": 0.0,
            "reasoning_score": 0.0,
            "format_score": 0.0,
            "overall_score": 0.0,
            "metadata": {
                "reason": reason
            },
        }

    @staticmethod
    def _interaction_explanation(
        claim_type: str,
        claimed_attack: Optional[str],
        claimed_attack_valid: bool,
        claimed_attack_remains: bool,
        real_attack_remains: bool,
    ) -> str:

        if claim_type == "no_attack":

            if real_attack_remains:
                return (
                    "Red claimed that no attack remains, "
                    "but ground truth indicates that a "
                    "real vulnerability remains."
                )

            return (
                "Red claimed that no attack remains and "
                "ground truth confirms that the defense "
                "closes the vulnerability."
            )

        if (
            claimed_attack_valid
            and claimed_attack_remains
        ):
            return (
                f"Red correctly identified "
                f"{claimed_attack} as a remaining "
                "exploitable attack."
            )

        if (
            not claimed_attack_valid
            and not real_attack_remains
        ):
            return (
                "Red claimed an attack that is not valid, "
                "and no real vulnerability remains."
            )

        if (
            not claimed_attack_valid
            and real_attack_remains
        ):
            return (
                "Red identified an incorrect attack while "
                "a different real vulnerability remains."
            )

        return (
            "Red's challenge could not be completely "
            "classified from deterministic evidence."
        )