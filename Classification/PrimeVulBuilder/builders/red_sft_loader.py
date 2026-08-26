import json
import re
from pathlib import Path
from typing import Any, Optional

from .models import RedExample


class RedSFTLoader:
    """
    Loads and analyzes red_sft.json.

    Purpose
    -------
    red_sft.json contains Red-Team analyses of a scenario.

    It is used by the PrimeVul builder as INTERNAL ground-truth
    and validation information.

    It is NOT model-visible.

    Important attack-label behavior
    -------------------------------
    A scenario may legitimately contain:

        Attack: none

    together with:

        Attack: <real vulnerability>

    The "none" examples are valid counterfactual / negative
    reasoning examples.

    A scenario may also contain multiple distinct non-"none"
    attack labels, particularly when the scenario represents
    multiple vulnerabilities or multiple valid attack paths.

    Therefore this loader DOES NOT require every Red SFT example
    to have the same Attack label.

    Instead:

        1. "none" is treated as a negative/counterfactual label.
        2. Non-"none" labels are treated as positive attack labels.
        3. Multiple positive attack labels are allowed.
        4. The loader provides helpers for obtaining:
               - all attack labels
               - positive attack labels
               - the primary/common attack when one exists
        5. Red SFT is never used as the sole source of the
           binary vulnerability target.

    It does NOT:
        - build prompts;
        - load source code;
        - load metadata.json;
        - read blue_sft.json;
        - expose Red reasoning to the model.
    """

    # ============================================================
    # CONSTANTS
    # ============================================================

    NONE_ATTACK_LABELS = {
        "none",
        "no attack",
        "no vulnerability",
        "not vulnerable",
        "non vulnerable",
        "non-vulnerable",
        "safe",
        "benign",
        "no exploitable vulnerability",
        "no exploitable vulnerabilities",
    }

    # ============================================================
    # ATTACK EXTRACTION
    # ============================================================

    ATTACK_PATTERN = re.compile(
        r"^\s*Attack\s*:\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # ============================================================
    # WRONG ACTIONS
    # ============================================================

    WRONG_ACTIONS_PATTERN = re.compile(
        r"^\s*wrong_actions\s*:\s*"
        r"(.*?)(?="
        r"^\s*wrong_action_reasoning\s*:\s*$"
        r"|"
        r"^\s*$"
        r"|"
        r"\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )

    # ============================================================
    # WRONG ACTION REASONING
    # ============================================================

    WRONG_REASONING_PATTERN = re.compile(
        r"^\s*wrong_action_reasoning\s*:\s*"
        r"(.*)$",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        logger=None,
        strict_attack_consistency: bool = False,
    ):
        """
        Parameters
        ----------
        logger:
            Optional logger instance.

        strict_attack_consistency:
            Retained for backward compatibility with the existing
            builder.

            IMPORTANT:
            Even when True, multiple valid attack labels do NOT
            automatically cause failure.

            The only consistency condition enforced by the loader
            is that the Red SFT file contains at least one valid
            Attack label.

            This is intentional because your dataset contains:

                - positive attack examples
                - negative "none" examples
                - multiple valid attack paths

            inside the same scenario.
        """

        self.logger = logger

        self.strict_attack_consistency = (
            strict_attack_consistency
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def load(
        self,
        red_sft_path: Path,
    ) -> list[RedExample]:
        """
        Load and validate all Red SFT examples from red_sft.json.
        """

        red_sft_path = Path(
            red_sft_path
        )

        if not red_sft_path.exists():
            raise FileNotFoundError(
                f"red_sft.json not found: "
                f"{red_sft_path}"
            )

        if not red_sft_path.is_file():
            raise ValueError(
                f"red_sft path is not a file: "
                f"{red_sft_path}"
            )

        data = self._read_json(
            red_sft_path
        )

        examples = self._parse_examples(
            data
        )

        if not examples:
            raise ValueError(
                f"No valid Red SFT examples found in: "
                f"{red_sft_path}"
            )

        self._validate_attack_consistency(
            examples,
            red_sft_path,
        )

        if self.logger:

            self.logger.info(
                f"Loaded {len(examples)} "
                f"Red SFT example(s)"
            )

            self._log_attack_summary(
                examples
            )

        return examples

    # ============================================================
    # JSON READING
    # ============================================================

    def _read_json(
        self,
        path: Path,
    ) -> Any:
        """
        Read red_sft.json as JSON.
        """

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(file)

        except json.JSONDecodeError as exc:

            raise ValueError(
                f"Invalid JSON in Red SFT file: "
                f"{path}\n{exc}"
            ) from exc

        except OSError as exc:

            raise OSError(
                f"Could not read Red SFT file: "
                f"{path}\n{exc}"
            ) from exc

    # ============================================================
    # EXAMPLE PARSING
    # ============================================================

    def _parse_examples(
        self,
        data: Any,
    ) -> list[RedExample]:
        """
        Parse the Red SFT JSON structure.

        Expected structure:

        [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "..."
                    },
                    {
                        "role": "user",
                        "content": "..."
                    },
                    {
                        "role": "assistant",
                        "content": "Attack: ..."
                    }
                ]
            }
        ]
        """

        if not isinstance(
            data,
            list,
        ):
            raise ValueError(
                "red_sft.json must contain "
                "a JSON list."
            )

        examples = []

        for index, item in enumerate(
            data,
            start=1,
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            messages = item.get(
                "messages"
            )

            if not isinstance(
                messages,
                list,
            ):
                continue

            assistant_content = (
                self._get_assistant_content(
                    messages
                )
            )

            if not assistant_content:
                continue

            question = (
                self._get_user_content(
                    messages
                )
            )

            attack = (
                self._extract_attack(
                    assistant_content
                )
            )

            reasoning = (
                self._extract_reasoning(
                    assistant_content
                )
            )

            wrong_actions = (
                self._extract_wrong_actions(
                    assistant_content
                )
            )

            wrong_action_reasoning = (
                self._extract_wrong_action_reasoning(
                    assistant_content
                )
            )

            example = RedExample(
                example_id=str(index),
                question=question,
                answer=assistant_content,
                attack=attack,
                reasoning=reasoning,
                wrong_actions=wrong_actions,
                wrong_action_reasoning=(
                    wrong_action_reasoning
                ),
            )

            examples.append(
                example
            )

        return examples

    # ============================================================
    # MESSAGE EXTRACTION
    # ============================================================

    def _get_assistant_content(
        self,
        messages: list[dict],
    ) -> Optional[str]:
        """
        Return the last assistant message.

        Using the last assistant message makes the loader robust
        if a future Red SFT example contains multiple assistant
        turns.
        """

        for message in reversed(
            messages
        ):

            if not isinstance(
                message,
                dict,
            ):
                continue

            if message.get(
                "role"
            ) != "assistant":
                continue

            content = message.get(
                "content"
            )

            if isinstance(
                content,
                str,
            ):

                content = content.strip()

                if content:
                    return content

        return None

    def _get_user_content(
        self,
        messages: list[dict],
    ) -> str:
        """
        Return the latest user question.
        """

        for message in reversed(
            messages
        ):

            if not isinstance(
                message,
                dict,
            ):
                continue

            if message.get(
                "role"
            ) != "user":
                continue

            content = message.get(
                "content"
            )

            if isinstance(
                content,
                str,
            ):
                return content.strip()

        return ""

    # ============================================================
    # ATTACK EXTRACTION
    # ============================================================

    def _extract_attack(
        self,
        assistant_content: str,
    ) -> Optional[str]:
        """
        Extract an Attack field.

        Examples:

            Attack: command_injection

        or:

            Attack: Settlement Allocation Context Drift

        or:

            Attack: none
        """

        match = self.ATTACK_PATTERN.search(
            assistant_content
        )

        if not match:
            return None

        attack = match.group(
            1
        ).strip()

        return attack or None

    # ============================================================
    # ATTACK CLASSIFICATION
    # ============================================================

    def is_none_attack(
        self,
        attack: Optional[str],
    ) -> bool:
        """
        Determine whether an Attack label represents a negative
        / counterfactual example.

        Examples:

            none
            no attack
            no vulnerability
            not vulnerable
            safe
            benign

        are treated as negative examples.
        """

        if not attack:
            return False

        normalized = (
            self._normalize_attack(
                attack
            )
        )

        return (
            normalized
            in self.NONE_ATTACK_LABELS
        )

    def is_positive_attack(
        self,
        attack: Optional[str],
    ) -> bool:
        """
        Return True when the attack label represents an actual
        vulnerability/attack rather than a negative example.
        """

        if not attack:
            return False

        return not self.is_none_attack(
            attack
        )

    # ============================================================
    # REASONING EXTRACTION
    # ============================================================

    def _extract_reasoning(
        self,
        assistant_content: str,
    ) -> Optional[str]:
        """
        Extract the main Red reasoning block.

        The reasoning is retained for validation/debugging only.

        It is NEVER inserted into the model classification
        prompt.
        """

        match = re.search(
            r"^\s*Reasoning\s*:\s*",
            assistant_content,
            re.IGNORECASE | re.MULTILINE,
        )

        if not match:
            return None

        reasoning = assistant_content[
            match.end():
        ].strip()

        # Stop before wrong_actions if present.
        wrong_actions_match = re.search(
            r"^\s*wrong_actions\s*:\s*",
            reasoning,
            re.IGNORECASE | re.MULTILINE,
        )

        if wrong_actions_match:

            reasoning = reasoning[
                :wrong_actions_match.start()
            ].strip()

        return (
            reasoning
            if reasoning
            else None
        )

    # ============================================================
    # WRONG ACTIONS
    # ============================================================

    def _extract_wrong_actions(
        self,
        assistant_content: str,
    ) -> list[str]:
        """
        Extract wrong_actions from the Red answer.

        Supports:

            wrong_actions:
            - jwt_forgery
            - sql_injection

        and:

            wrong_actions:
            jwt_forgery
            sql_injection
        """

        match = re.search(
            r"^\s*wrong_actions\s*:\s*$"
            r"(.*?)"
            r"(?="
            r"^\s*wrong_action_reasoning\s*:\s*$"
            r"|"
            r"\Z)",
            assistant_content,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )

        if not match:
            return []

        block = match.group(
            1
        )

        actions = []

        for line in block.splitlines():

            line = line.strip()

            if not line:
                continue

            # Remove Markdown list markers.
            line = re.sub(
                r"^[-*+]\s*",
                "",
                line,
            ).strip()

            if not line:
                continue

            actions.append(
                line
            )

        return actions

    # ============================================================
    # WRONG ACTION REASONING
    # ============================================================

    def _extract_wrong_action_reasoning(
        self,
        assistant_content: str,
    ) -> Optional[str]:
        """
        Extract wrong_action_reasoning.

        This is retained only for internal analysis/debugging.
        """

        match = re.search(
            r"^\s*wrong_action_reasoning\s*:\s*"
            r"(.*)$",
            assistant_content,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )

        if not match:
            return None

        reasoning = match.group(
            1
        ).strip()

        return (
            reasoning
            if reasoning
            else None
        )

    # ============================================================
    # ATTACK VALIDATION
    # ============================================================

    def _validate_attack_consistency(
        self,
        examples: list[RedExample],
        red_sft_path: Path,
    ) -> None:
        """
        Validate the Attack fields.

        IMPORTANT
        ---------
        We no longer require a single common attack.

        Valid examples include:

            none
            real_attack

        and:

            real_attack_1
            real_attack_2
            real_attack_3

        The loader only requires at least one valid Attack field.

        A file containing ONLY "none" examples is still valid as
        Red SFT data, because those are legitimate negative/
        counterfactual examples.

        This method therefore performs structural validation,
        not single-label consistency enforcement.
        """

        attacks = [
            example.attack
            for example in examples
            if example.attack
        ]

        if not attacks:

            raise ValueError(
                f"No Attack label found in Red SFT file: "
                f"{red_sft_path}"
            )

        positive_attacks = [
            attack
            for attack in attacks
            if self.is_positive_attack(
                attack
            )
        ]

        negative_attacks = [
            attack
            for attack in attacks
            if self.is_none_attack(
                attack
            )
        ]

        # --------------------------------------------------------
        # Log useful diagnostic information.
        # --------------------------------------------------------

        if self.logger:

            self.logger.info(
                f"Red SFT labels: "
                f"{len(attacks)} total, "
                f"{len(positive_attacks)} positive, "
                f"{len(negative_attacks)} negative"
            )

        # --------------------------------------------------------
        # Multiple positive attack labels are valid.
        #
        # We intentionally DO NOT raise here.
        # --------------------------------------------------------

        unique_positive = (
            self.get_unique_positive_attacks(
                examples
            )
        )

        if (
            len(unique_positive) > 1
            and self.logger
        ):

            self.logger.info(
                "Multiple valid Red SFT attack labels "
                f"detected: {unique_positive}"
            )

        # --------------------------------------------------------
        # Mixed positive/negative examples are valid.
        # --------------------------------------------------------

        if (
            positive_attacks
            and negative_attacks
            and self.logger
        ):

            self.logger.info(
                "Red SFT contains both positive attack "
                "and negative/counterfactual examples; "
                "this is valid."
            )

    # ============================================================
    # LOG ATTACK SUMMARY
    # ============================================================

    def _log_attack_summary(
        self,
        examples: list[RedExample],
    ) -> None:
        """
        Log a readable summary without exposing Red reasoning.
        """

        positive = (
            self.get_unique_positive_attacks(
                examples
            )
        )

        negative_count = sum(
            1
            for example in examples
            if self.is_none_attack(
                example.attack
            )
        )

        if positive:

            self.logger.info(
                f"Positive attack label(s): "
                f"{positive}"
            )

        if negative_count:

            self.logger.info(
                f"Negative/counterfactual examples: "
                f"{negative_count}"
            )

    # ============================================================
    # COMMON ATTACK
    # ============================================================

    def get_common_attack(
        self,
        examples: list[RedExample],
    ) -> Optional[str]:
        """
        Return the attack label when there is exactly ONE unique
        positive attack.

        Behavior
        --------

        One positive attack:

            [
                none,
                safety_configuration_lineage_drift,
                safety_configuration_lineage_drift
            ]

            -> "safety_configuration_lineage_drift"

        Multiple positive attacks:

            [
                attack_a,
                attack_b,
                attack_c
            ]

            -> None

        Only negative examples:

            [
                none,
                none
            ]

            -> "none"

        This avoids falsely claiming that one arbitrary attack is
        the "common" attack when multiple distinct attacks exist.
        """

        attacks = [
            example.attack
            for example in examples
            if example.attack
        ]

        if not attacks:
            return None

        positive_attacks = [
            attack
            for attack in attacks
            if self.is_positive_attack(
                attack
            )
        ]

        # --------------------------------------------------------
        # No positive attack.
        # --------------------------------------------------------

        if not positive_attacks:

            return "none"

        # --------------------------------------------------------
        # Find unique positive labels.
        # --------------------------------------------------------

        groups = {}

        for attack in positive_attacks:

            key = self._normalize_attack(
                attack
            )

            groups.setdefault(
                key,
                [],
            ).append(attack)

        # --------------------------------------------------------
        # Exactly one positive attack.
        # --------------------------------------------------------

        if len(groups) == 1:

            return positive_attacks[0]

        # --------------------------------------------------------
        # Multiple distinct positive attacks.
        #
        # There is no single common attack.
        # --------------------------------------------------------

        return None

    # ============================================================
    # ALL ATTACK LABELS
    # ============================================================

    def get_all_attacks(
        self,
        examples: list[RedExample],
    ) -> list[str]:
        """
        Return all unique attack labels, including "none".
        """

        seen = set()
        result = []

        for example in examples:

            attack = example.attack

            if not attack:
                continue

            normalized = (
                self._normalize_attack(
                    attack
                )
            )

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            result.append(
                attack
            )

        return result

    # ============================================================
    # POSITIVE ATTACKS
    # ============================================================

    def get_positive_attacks(
        self,
        examples: list[RedExample],
    ) -> list[str]:
        """
        Return all unique positive/non-"none" attack labels.

        Example:

            none
            safety_configuration_lineage_drift
            safety_configuration_lineage_drift

        returns:

            [
                "safety_configuration_lineage_drift"
            ]
        """

        return [
            attack
            for attack in self.get_all_attacks(
                examples
            )
            if self.is_positive_attack(
                attack
            )
        ]

    # ============================================================
    # UNIQUE POSITIVE ATTACKS
    # ============================================================

    def get_unique_positive_attacks(
        self,
        examples: list[RedExample],
    ) -> list[str]:
        """
        Return unique positive attack labels using normalized
        comparison while preserving the first observed spelling.
        """

        groups = {}

        for example in examples:

            attack = example.attack

            if not self.is_positive_attack(
                attack
            ):
                continue

            normalized = (
                self._normalize_attack(
                    attack
                )
            )

            if normalized not in groups:

                groups[
                    normalized
                ] = attack

        return list(
            groups.values()
        )

    # ============================================================
    # GROUND-TRUTH ATTACK
    # ============================================================

    def get_attack(
        self,
        examples: list[RedExample],
    ) -> str:
        """
        Return the Red SFT attack when there is exactly one
        positive attack.

        If multiple positive attacks exist, raise an explicit
        error rather than silently selecting one.

        Why?
        ----
        A scenario such as 090 or 095 legitimately contains
        multiple attack labels. Choosing the first one would
        create false ground truth.

        The caller should instead use:

            get_positive_attacks()

        when multiple attack paths are expected.
        """

        attack = self.get_common_attack(
            examples
        )

        if attack is not None:

            return attack

        positive_attacks = (
            self.get_unique_positive_attacks(
                examples
            )
        )

        raise ValueError(
            "Red SFT contains multiple distinct "
            "positive attack labels. Use "
            "get_positive_attacks() instead. "
            f"Attacks: {positive_attacks}"
        )

    # ============================================================
    # REASONING
    # ============================================================

    def get_all_reasoning(
        self,
        examples: list[RedExample],
    ) -> list[str]:
        """
        Return all available Red reasoning blocks.

        Used only for debugging/validation.

        NEVER insert these into the model classification prompt.
        """

        return [
            example.reasoning
            for example in examples
            if example.reasoning
        ]

    # ============================================================
    # WRONG ACTIONS
    # ============================================================

    def get_all_wrong_actions(
        self,
        examples: list[RedExample],
    ) -> list[str]:
        """
        Return the unique union of wrong actions across
        all Red examples.
        """

        actions = set()

        for example in examples:

            for action in (
                example.wrong_actions
            ):
                actions.add(
                    action
                )

        return sorted(
            actions
        )

    # ============================================================
    # METADATA VALIDATION
    # ============================================================

    def validate_against_attack(
        self,
        examples: list[RedExample],
        expected_attack: Optional[str],
    ) -> bool:
        """
        Compare Red-derived attacks against an expected attack
        from metadata.json.

        IMPORTANT
        ---------
        Multiple Red attack labels are allowed.

        Validation succeeds when the expected metadata attack
        matches ANY positive Red SFT attack label.

        Example:

            metadata:
                optimal_attack =
                "Safety Configuration Lineage Drift"

            Red SFT:
                none
                safety_configuration_lineage_drift

        -> True

        Example:

            metadata:
                optimal_attack =
                "Capability Authorization Merge Abuse"

            Red SFT:
                attack_a
                attack_b

        -> True only if one of the positive labels matches
           the metadata attack.

        If metadata does not provide an expected attack,
        validation passes.
        """

        if not expected_attack:
            return True

        expected_normalized = (
            self._normalize_attack(
                expected_attack
            )
        )

        positive_attacks = (
            self.get_unique_positive_attacks(
                examples
            )
        )

        positive_normalized = {
            self._normalize_attack(
                attack
            )
            for attack in positive_attacks
        }

        return (
            expected_normalized
            in positive_normalized
        )

    # ============================================================
    # NEGATIVE EXAMPLES
    # ============================================================

    def get_negative_examples(
        self,
        examples: list[RedExample],
    ) -> list[RedExample]:
        """
        Return Red SFT examples whose attack label is "none"
        or another recognized negative state.
        """

        return [
            example
            for example in examples
            if self.is_none_attack(
                example.attack
            )
        ]

    # ============================================================
    # POSITIVE EXAMPLES
    # ============================================================

    def get_positive_examples(
        self,
        examples: list[RedExample],
    ) -> list[RedExample]:
        """
        Return Red SFT examples containing actual attack labels.
        """

        return [
            example
            for example in examples
            if self.is_positive_attack(
                example.attack
            )
        ]

    # ============================================================
    # DEBUG SUMMARY
    # ============================================================

    def summary(
        self,
        examples: list[RedExample],
    ) -> dict:
        """
        Produce a compact internal summary.

        This information must NOT be inserted into the model
        prompt.
        """

        positive_attacks = (
            self.get_unique_positive_attacks(
                examples
            )
        )

        all_attacks = (
            self.get_all_attacks(
                examples
            )
        )

        negative_examples = (
            self.get_negative_examples(
                examples
            )
        )

        positive_examples = (
            self.get_positive_examples(
                examples
            )
        )

        return {
            "num_examples": len(
                examples
            ),

            "all_attack_labels": (
                all_attacks
            ),

            "positive_attack_labels": (
                positive_attacks
            ),

            "negative_example_count": (
                len(
                    negative_examples
                )
            ),

            "positive_example_count": (
                len(
                    positive_examples
                )
            ),

            "common_attack": (
                self.get_common_attack(
                    examples
                )
            ),

            "reasoning_examples": len(
                self.get_all_reasoning(
                    examples
                )
            ),

            "wrong_actions": (
                self.get_all_wrong_actions(
                    examples
                )
            ),
        }

    # ============================================================
    # ATTACK NORMALIZATION
    # ============================================================

    def _normalize_attack(
        self,
        attack: str,
    ) -> str:
        """
        Normalize attack labels for comparison.

        Examples considered equivalent:

            Command Injection
            command_injection
            command-injection
            command injection
        """

        normalized = (
            attack
            .lower()
            .strip()
        )

        normalized = re.sub(
            r"[_\-]+",
            " ",
            normalized,
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized