from typing import Optional


class ContextEstimator:
    """
    Estimates the size of generated classification prompts.

    Purpose
    -------
    The PrimeVul builder can combine:

        system prompt
        +
        scenario context
        +
        multiple source-code files

    Cross-file scenarios can therefore produce very large
    prompts.

    This class estimates prompt size before the final dataset
    is generated.

    Important
    ---------
    Character-based token counts are only approximations.

    For exact counts, provide the actual tokenizer used by the
    model through estimate_with_tokenizer().

    This class does NOT:
        - truncate source code;
        - select files;
        - modify prompts;
        - determine vulnerability labels;
        - determine target = 0/1.
    """

    # ============================================================
    # DEFAULT SETTINGS
    # ============================================================

    # Rough approximation for ordinary English/source code.
    #
    # This is NOT an exact tokenizer conversion.
    DEFAULT_CHARS_PER_TOKEN = 4.0

    # Prompt-size warning threshold.
    DEFAULT_WARNING_TOKENS = 12000

    # Prompt-size hard reporting threshold.
    #
    # The estimator only reports this condition.
    # It does NOT truncate the prompt.
    DEFAULT_LIMIT_TOKENS = 16000

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        logger=None,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
        warning_tokens: int = DEFAULT_WARNING_TOKENS,
        limit_tokens: int = DEFAULT_LIMIT_TOKENS,
    ):
        """
        Parameters
        ----------
        logger:
            Optional Logger instance.

        chars_per_token:
            Approximate number of characters per token.

        warning_tokens:
            Token count at which a warning is generated.

        limit_tokens:
            Token count at which the prompt is reported as
            OVER_LIMIT.
        """

        if chars_per_token <= 0:
            raise ValueError(
                "chars_per_token must be greater than zero."
            )

        if warning_tokens <= 0:
            raise ValueError(
                "warning_tokens must be greater than zero."
            )

        if limit_tokens <= 0:
            raise ValueError(
                "limit_tokens must be greater than zero."
            )

        if warning_tokens > limit_tokens:
            raise ValueError(
                "warning_tokens cannot be greater than "
                "limit_tokens."
            )

        self.logger = logger

        self.chars_per_token = (
            chars_per_token
        )

        self.warning_tokens = (
            warning_tokens
        )

        self.limit_tokens = (
            limit_tokens
        )

    # ============================================================
    # BASIC CHARACTER ESTIMATION
    # ============================================================

    def estimate_characters(
        self,
        text: Optional[str],
    ) -> int:
        """
        Return the number of characters in text.
        """

        if not text:
            return 0

        return len(text)

    # ============================================================
    # APPROXIMATE TOKEN ESTIMATION
    # ============================================================

    def estimate_tokens(
        self,
        text: Optional[str],
    ) -> int:
        """
        Estimate token count from character count.

        This is an approximation only.

        For exact model-specific token counts use:

            estimate_with_tokenizer()
        """

        if not text:
            return 0

        characters = len(text)

        return max(
            1,
            round(
                characters
                / self.chars_per_token
            ),
        )

    # ============================================================
    # ACTUAL TOKENIZER ESTIMATION
    # ============================================================

    def estimate_with_tokenizer(
        self,
        text: Optional[str],
        tokenizer,
    ) -> int:
        """
        Estimate tokens using an actual Hugging Face-style
        tokenizer.

        Example:

            tokenizer(
                text,
                add_special_tokens=True,
                return_attention_mask=False,
            )

        The tokenizer is injected instead of imported so that
        this utility does not require transformers as a direct
        dependency.
        """

        if not text:
            return 0

        if tokenizer is None:
            raise ValueError(
                "A tokenizer is required for exact "
                "token estimation."
            )

        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
        )

        input_ids = encoded.get(
            "input_ids"
        )

        if input_ids is None:
            raise ValueError(
                "Tokenizer did not return input_ids."
            )

        return len(
            input_ids
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
        estimated_tokens: int,
    ) -> str:
        """
        Return the prompt-size status.

        Possible values:

            OK
            WARNING
            OVER_LIMIT
        """

        if estimated_tokens >= self.limit_tokens:
            return "OVER_LIMIT"

        if estimated_tokens >= self.warning_tokens:
            return "WARNING"

        return "OK"

    # ============================================================
    # BASIC TEXT STATISTICS
    # ============================================================

    def estimate_text(
        self,
        text: Optional[str],
    ) -> dict:
        """
        Return character and approximate-token statistics for
        one text block.
        """

        characters = (
            self.estimate_characters(
                text
            )
        )

        estimated_tokens = (
            self.estimate_tokens(
                text
            )
        )

        return {
            "characters": characters,
            "estimated_tokens": estimated_tokens,
            "status": self.get_status(
                estimated_tokens
            ),
        }

    # ============================================================
    # COMPLETE PROMPT ESTIMATION
    # ============================================================

    def estimate_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """
        Estimate the size of the complete model input.

        The model input consists of:

            system_prompt
            +
            user_prompt
        """

        system_characters = (
            self.estimate_characters(
                system_prompt
            )
        )

        user_characters = (
            self.estimate_characters(
                user_prompt
            )
        )

        total_characters = (
            system_characters
            + user_characters
        )

        system_tokens = (
            self.estimate_tokens(
                system_prompt
            )
        )

        user_tokens = (
            self.estimate_tokens(
                user_prompt
            )
        )

        total_tokens = (
            system_tokens
            + user_tokens
        )

        return {
            "system": {
                "characters": system_characters,
                "estimated_tokens": system_tokens,
            },

            "user": {
                "characters": user_characters,
                "estimated_tokens": user_tokens,
            },

            "total": {
                "characters": total_characters,
                "estimated_tokens": total_tokens,
            },

            "status": self.get_status(
                total_tokens
            ),

            "warning_threshold": (
                self.warning_tokens
            ),

            "limit_threshold": (
                self.limit_tokens
            ),
        }

    # ============================================================
    # COMPONENT-LEVEL ESTIMATION
    # ============================================================

    def estimate_components(
        self,
        system_prompt: str,
        scenario_context: str,
        source_code: str,
    ) -> dict:
        """
        Estimate each major prompt component independently.

        This is useful for finding which component makes a
        scenario too large.
        """

        system_characters = (
            self.estimate_characters(
                system_prompt
            )
        )

        scenario_characters = (
            self.estimate_characters(
                scenario_context
            )
        )

        source_characters = (
            self.estimate_characters(
                source_code
            )
        )

        system_tokens = (
            self.estimate_tokens(
                system_prompt
            )
        )

        scenario_tokens = (
            self.estimate_tokens(
                scenario_context
            )
        )

        source_tokens = (
            self.estimate_tokens(
                source_code
            )
        )

        total_characters = (
            system_characters
            + scenario_characters
            + source_characters
        )

        total_tokens = (
            system_tokens
            + scenario_tokens
            + source_tokens
        )

        return {
            "system": {
                "characters": system_characters,
                "estimated_tokens": system_tokens,
            },

            "scenario": {
                "characters": scenario_characters,
                "estimated_tokens": scenario_tokens,
            },

            "source_code": {
                "characters": source_characters,
                "estimated_tokens": source_tokens,
            },

            "total": {
                "characters": total_characters,
                "estimated_tokens": total_tokens,
            },

            "status": self.get_status(
                total_tokens
            ),

            "warning_threshold": (
                self.warning_tokens
            ),

            "limit_threshold": (
                self.limit_tokens
            ),
        }

    # ============================================================
    # SOURCE CODE ESTIMATION
    # ============================================================

    def estimate_source_code(
        self,
        source_code: str,
    ) -> dict:
        """
        Estimate the size of the source-code portion.
        """

        return self.estimate_text(
            source_code
        )

    # ============================================================
    # SCENARIO CONTEXT ESTIMATION
    # ============================================================

    def estimate_context(
        self,
        scenario_context: str,
    ) -> dict:
        """
        Estimate the size of the neutral scenario context.
        """

        return self.estimate_text(
            scenario_context
        )

    # ============================================================
    # TOKEN BUDGET
    # ============================================================

    def fits_budget(
        self,
        estimated_tokens: int,
        max_tokens: int,
    ) -> bool:
        """
        Return True if the estimated input fits within the
        supplied token budget.
        """

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        return (
            estimated_tokens
            <= max_tokens
        )

    # ============================================================
    # REMAINING BUDGET
    # ============================================================

    def remaining_budget(
        self,
        estimated_tokens: int,
        max_tokens: int,
    ) -> int:
        """
        Return the remaining token budget.

        Negative value means the estimated input exceeds the
        supplied budget.
        """

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        return (
            max_tokens
            - estimated_tokens
        )

    # ============================================================
    # LOGGER
    # ============================================================

    def log_statistics(
        self,
        statistics: dict,
        scenario_name: Optional[str] = None,
    ) -> None:
        """
        Log prompt-size statistics if a logger is available.
        """

        if self.logger is None:
            return

        prefix = ""

        if scenario_name:
            prefix = (
                f"[{scenario_name}] "
            )

        total = statistics.get(
            "total",
            {},
        )

        estimated_tokens = total.get(
            "estimated_tokens",
            0,
        )

        characters = total.get(
            "characters",
            0,
        )

        status = statistics.get(
            "status",
            "UNKNOWN",
        )

        self.logger.info(
            f"{prefix}"
            f"Prompt size: "
            f"{estimated_tokens:,} estimated tokens | "
            f"{characters:,} characters | "
            f"status={status}"
        )

    # ============================================================
    # BUILD REPORT
    # ============================================================

    def build_report(
        self,
        system_prompt: str,
        scenario_context: str,
        source_code: str,
        scenario_name: Optional[str] = None,
    ) -> dict:
        """
        Build a detailed prompt-size report for one scenario.
        """

        report = self.estimate_components(
            system_prompt=system_prompt,
            scenario_context=scenario_context,
            source_code=source_code,
        )

        report["scenario_name"] = (
            scenario_name
        )

        return report