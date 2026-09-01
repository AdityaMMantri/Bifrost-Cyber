"""
context_estimator.py

Estimates prompt/token size before generation.

Used to prevent prompts from exceeding the configured
model context window.

This module does not:
    - modify prompts
    - remove files
    - select relevant/noise files
    - build prompts
    - generate responses
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ImportError:
    torch = None


@dataclass
class ContextEstimate:
    """Estimated context size."""

    character_count: int
    estimated_tokens: int
    max_input_tokens: int
    fits: bool
    overflow_tokens: int

    @property
    def utilization(self) -> float:
        """Fraction of the available context being used."""

        if self.max_input_tokens <= 0:
            return 0.0

        return self.estimated_tokens / self.max_input_tokens


class ContextEstimator:
    """
    Estimate token usage for prompts.

    Preferred method:
        use the actual tokenizer.

    Fallback:
        use a character-based approximation.

    The estimator is intentionally conservative enough
    for deciding whether a prompt needs attention.
    """

    def __init__(
        self,
        tokenizer=None,
        max_input_tokens: int = 12288,
        chars_per_token: float = 4.0,
    ):
        if max_input_tokens <= 0:
            raise ValueError(
                "max_input_tokens must be greater than 0."
            )

        if chars_per_token <= 0:
            raise ValueError(
                "chars_per_token must be greater than 0."
            )

        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.chars_per_token = chars_per_token

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def estimate(
        self,
        text: str,
    ) -> ContextEstimate:
        """
        Estimate the token count of a prompt.
        """

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )

        character_count = len(text)

        if self.tokenizer is not None:
            estimated_tokens = self._tokenizer_count(text)
        else:
            estimated_tokens = self._character_estimate(
                character_count
            )

        overflow = max(
            0,
            estimated_tokens - self.max_input_tokens,
        )

        return ContextEstimate(
            character_count=character_count,
            estimated_tokens=estimated_tokens,
            max_input_tokens=self.max_input_tokens,
            fits=estimated_tokens <= self.max_input_tokens,
            overflow_tokens=overflow,
        )

    # ------------------------------------------------------------------
    # Tokenizer-based estimation
    # ------------------------------------------------------------------

    def _tokenizer_count(
        self,
        text: str,
    ) -> int:
        """Count tokens using the model tokenizer."""

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )

        input_ids = encoded["input_ids"]

        if hasattr(input_ids, "shape"):
            return int(input_ids.shape[-1])

        return len(input_ids)

    # ------------------------------------------------------------------
    # Character approximation
    # ------------------------------------------------------------------

    def _character_estimate(
        self,
        character_count: int,
    ) -> int:
        """
        Estimate tokens without loading a tokenizer.

        This is only an approximation.
        """

        return max(
            1,
            int(
                character_count
                / self.chars_per_token
            ),
        )

    # ------------------------------------------------------------------
    # Convenience checks
    # ------------------------------------------------------------------

    def fits(
        self,
        text: str,
    ) -> bool:
        """Return True if the prompt fits the context limit."""

        return self.estimate(text).fits

    def token_count(
        self,
        text: str,
    ) -> int:
        """Return the estimated token count."""

        return self.estimate(
            text
        ).estimated_tokens

    def remaining_tokens(
        self,
        text: str,
    ) -> int:
        """
        Return remaining input-token capacity.

        Returns 0 when the prompt already exceeds the limit.
        """

        estimate = self.estimate(text)

        return max(
            0,
            estimate.max_input_tokens
            - estimate.estimated_tokens,
        )

    def exceeds_limit(
        self,
        text: str,
    ) -> bool:
        """Return True when the prompt exceeds the context limit."""

        return not self.fits(text)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        text: str,
    ) -> ContextEstimate:
        """
        Estimate and raise an error if the prompt is too large.

        Useful before generation when exceeding the context
        window should be treated as a hard error.
        """

        result = self.estimate(text)

        if not result.fits:
            raise ValueError(
                "Prompt exceeds the configured context limit: "
                f"{result.estimated_tokens} tokens > "
                f"{result.max_input_tokens} tokens "
                f"(overflow={result.overflow_tokens})."
            )

        return result