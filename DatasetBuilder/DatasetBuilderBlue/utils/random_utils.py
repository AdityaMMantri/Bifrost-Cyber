"""
random_utils.py

Utility functions for reproducible random operations used
throughout the Dataset Builder.
"""

import random

from config import RANDOM_SEED


class RandomUtils:

    def __init__(self, seed=None):

        if seed is None:
            seed = RANDOM_SEED

        self.random = random.Random(seed)

    # ======================================================
    # Random Choice
    # ======================================================

    def choice(self, items):

        if not items:
            raise ValueError(
                "Cannot choose from an empty collection."
            )

        return self.random.choice(items)

    # ======================================================
    # Shuffle
    # ======================================================

    def shuffle(self, items):

        copied = list(items)

        self.random.shuffle(copied)

        return copied

    # ======================================================
    # Sample
    # ======================================================

    def sample(self, items, k):

        if not items:
            return []

        k = max(0, min(k, len(items)))

        return self.random.sample(items, k)

    # ======================================================
    # Random Integer
    # ======================================================

    def randint(self, minimum, maximum):

        return self.random.randint(
            minimum,
            maximum
        )

    # ======================================================
    # Select Noise Files
    # ======================================================

    def select_noise_files(
        self,
        noise_files,
        minimum,
        maximum
    ):

        if not noise_files:
            return []

        count = self.randint(
            minimum,
            maximum
        )

        return self.sample(
            noise_files,
            count
        )


# ==========================================================
# Global Instance
# ==========================================================

random_utils = RandomUtils()