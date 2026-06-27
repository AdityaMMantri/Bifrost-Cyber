"""
Rough prompt statistics.

This is NOT tokenizer specific.
"""

class ContextEstimator:

    CHARS_PER_TOKEN = 4

    @staticmethod
    def estimate_tokens(text):

        if not text:

            return 0

        return len(text) // ContextEstimator.CHARS_PER_TOKEN

    @staticmethod
    def summary(prompt):

        chars = len(prompt)

        tokens = ContextEstimator.estimate_tokens(prompt)

        return {

            "characters": chars,

            "estimated_tokens": tokens

        }