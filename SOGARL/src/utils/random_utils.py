"""
random_utils.py

Shared randomization utilities for SOGARL.

This module provides reproducible random operations used throughout
the training pipeline.

Responsibilities
----------------

    - random seeding
    - random integer generation
    - random float generation
    - random choice
    - random sampling without replacement
    - shuffling
    - random train/validation/test splitting
    - selecting a random number of items from a range

This module does NOT:

    - select relevant files
    - decide which categories are weak
    - perform weakness mining
    - build prompts
    - generate model responses
    - calculate Oracle rewards
    - perform GRPO
    - load models

Important
---------

SOGARL uses randomness in several places.

For example:

    PathResolver
        ↓
    select 2–5 noise files

    Scenario sampler
        ↓
    shuffle scenarios

    WeaknessSampler
        ↓
    choose weak/random category

    Generation
        ↓
    model sampling

Model generation randomness is controlled separately through
the model/generation configuration.

This module controls Python/NumPy-level randomization.
"""


from __future__ import annotations
import random
from typing import List,Optional,Sequence,Tuple,TypeVar
import numpy as np

T = TypeVar("T")
class RandomUtils:
    """
    Stateless helper class for reproducible random operations.

    A local random.Random instance is used rather than relying
    exclusively on the global Python random state.

    This makes it possible for SOGARL components to maintain
    controlled randomness without unnecessarily interfering
    with one another.
    """

    def __init__(self,seed: Optional[int] = None) -> None:
        self.seed = seed
        self._random = random.Random(seed)
    # ========================================================================
    # SEED
    # ========================================================================
    def set_seed(self,seed: int) -> None:
        """
        Reset the local random generator.

        Example:

            random_utils.set_seed(42)
        """
        self.seed = seed
        self._random.seed(seed)
    # ========================================================================
    # INTEGER
    # ========================================================================
    def randint(self,minimum: int,maximum: int) -> int:
        """
        Return a random integer in the inclusive range:

            minimum <= value <= maximum
        """
        if minimum>maximum:
            raise ValueError("minimum cannot be greater than maximum.")
        return self._random.randint(minimum,maximum)
    # ========================================================================
    # FLOAT
    # ========================================================================
    def random(self) -> float:
        """
        Return a random float in:

            [0.0, 1.0)
        """
        return self._random.random()

    # ========================================================================
    # CHOICE
    # ========================================================================
    def choice(self,items: Sequence[T]) -> T:
        """
        Select one random item from a non-empty sequence.
        """
        if not items:
            raise ValueError("Cannot choose from an empty sequence.")
        return self._random.choice(items)
    # ========================================================================
    # SAMPLE
    # ========================================================================
    def sample(self,items: Sequence[T],count: int) -> List[T]:
        """
        Select `count` unique items without replacement.

        This is the main operation used for selecting noise files.

        Example:

            sample(noise_files, 3)

        """

        if count < 0:
            raise ValueError("count cannot be negative.")
        if count > len(items):
            raise ValueError(
                f"Cannot sample {count} items "
                f"from a collection containing "
                f"only {len(items)} items.")
        return self._random.sample(list(items),count)
    # ========================================================================
    # SHUFFLE
    # ========================================================================

    def shuffle(self,items: Sequence[T]) -> List[T]:
        """
        Return a shuffled copy of the supplied sequence.

        The original sequence is not modified.
        """
        shuffled = list(items)
        self._random.shuffle(shuffled)
        return shuffled

    # ========================================================================
    # RANDOM NUMBER OF ITEMS
    # ========================================================================
    def random_count(self,minimum: int,maximum: int) -> int:
        """
        Return a random integer between minimum and maximum.

        Used when the number of selected noise files should
        vary between episodes.

        Example:

            random_count(2, 5)

        may return:

            2
            3
            4
            5
        """
        return self.randint(minimum,maximum)

    # ========================================================================
    # RANDOM SAMPLE WITHIN RANGE
    # ========================================================================

    def sample_count_range(self,items: Sequence[T],minimum_count: int,maximum_count: int) -> List[T]:
        """
        Select a random number of unique items from a sequence.

        The number selected is randomly chosen between:

            minimum_count
            and
            maximum_count

        inclusive.

        This is useful for SOGARL's:

            2–5 noise files

        rule.
        """
        if minimum_count < 0:
            raise ValueError("minimum_count cannot be negative.")
        if maximum_count < minimum_count:
            raise ValueError(
                "maximum_count cannot be less "
                "than minimum_count.")
        if maximum_count > len(items):
            raise ValueError(
                f"maximum_count={maximum_count} "
                f"exceeds available items={len(items)}.")
        count = self.random_count(minimum_count,maximum_count)
        return self.sample(items,count)
    # ========================================================================
    # WEIGHTED CHOICE
    # ========================================================================

    def weighted_choice(self,items: Sequence[T],weights: Sequence[float]) -> T:
        """
        Select one item according to supplied probabilities/weights.

        Example:

            items   = ["weak", "random"]
            weights = [0.70, 0.30]

        This is useful for the weakness sampler.

        The method itself does not know what the items mean.
        """
        if not items:
            raise ValueError("Cannot choose from an empty sequence.")
        if len(items) != len(weights):
            raise ValueError(
                "items and weights must have "
                "the same length."
            )
        if any(weight < 0 for weight in weights):
            raise ValueError("Weights cannot be negative.")
        if sum(weights) <= 0:
            raise ValueError("At least one weight must be positive.")
        
        return self._random.choices(list(items),weights=list(weights),k=1)[0]

    # ========================================================================
    # PROBABILITY CHECK
    # ========================================================================

    def probability(self,probability: float) -> bool:
        """
        Return True with the supplied probability.

        Example:

            probability(0.70)

        returns True approximately 70% of the time.

        Useful for probabilistic sampling decisions.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "probability must be between "
                "0.0 and 1.0."
            )

        return (self.random() < probability)

    # ========================================================================
    # TRAIN / VALIDATION / TEST SPLIT
    # ========================================================================

    def train_validation_test_split(self,items: Sequence[T],train_ratio: float,validation_ratio: float,test_ratio: float) -> Tuple[
        List[T],
        List[T],
        List[T]]:
        """
        Randomly split items into:

            train
            validation
            test

        The supplied ratios must sum to 1.

        Example:

            0.70 / 0.15 / 0.15
        """

        ratios = (train_ratio,validation_ratio,test_ratio)

        if any(ratio < 0 for ratio in ratios):
            raise ValueError("Split ratios cannot be negative.")
        total = sum(ratios)
        if not np.isclose(total,1.0,atol=1e-10):
            raise ValueError(
                "train_ratio + validation_ratio "
                "+ test_ratio must equal 1.0.")
        shuffled = self.shuffle(items)
        total_items = len(shuffled)
        train_end = int(total_items*train_ratio)
        validation_end = (train_end + int(total_items*validation_ratio))
        train_items = shuffled[:train_end]
        validation_items = shuffled[train_end:validation_end]
        test_items = shuffled[validation_end:]

        return (
            train_items,
            validation_items,
            test_items,
        )

    # ========================================================================
    # NUMPY SEED
    # ========================================================================

    def seed_numpy(self) -> None:
        """
        Seed NumPy using the configured seed.

        This is useful when a component uses NumPy randomness
        in addition to Python's random module.
        """
        if self.seed is None:
            return
        np.random.seed(self.seed)

    # ========================================================================
    # PYTHON GLOBAL SEED
    # ========================================================================

    def seed_python(self) -> None:
        """
        Seed Python's global random generator.

        Normally the local generator should be preferred.

        This method exists for compatibility with code or
        third-party components that use Python's global
        random module.
        """
        if self.seed is None:
            return
        random.seed(self.seed)
    # ========================================================================
    # GLOBAL REPRODUCIBILITY
    # ========================================================================

    def seed_all(self) -> None:
        """
        Seed Python and NumPy random generators.

        PyTorch seeding is intentionally NOT performed here.

        PyTorch/CUDA reproducibility belongs to:

            utils/seed.py

        because that module handles:

            - torch.manual_seed()
            - CUDA seeds
            - deterministic settings
            - CUDA-specific behavior
        """
        self.seed_python()
        self.seed_numpy()