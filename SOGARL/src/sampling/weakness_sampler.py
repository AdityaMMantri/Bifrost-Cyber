"""
weakness_sampler.py

Adaptive scenario sampler for SOGARL.

Uses historical failure rates to focus training on weak
attack categories while retaining random exploration.

Sampling policy:

    70% → weakness-focused sampling
    30% → random sampling

The sampler does not:
    - evaluate model responses
    - calculate rewards
    - calculate advantages
    - update models
    - perform GRPO
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


class WeaknessSampler:

    def __init__(
        self,
        scenarios: Iterable[Any],
        weak_category_probability: float = 0.70,
        random_category_probability: float = 0.30,
        min_episodes: int = 20,
        rolling_window: int = 100,
        use_failure_rate: bool = True,
        seed: Optional[int] = 42,
        logger=None,
    ):
        self.scenarios = list(scenarios)

        if not self.scenarios:
            raise ValueError(
                "WeaknessSampler requires at least one scenario."
            )

        if not 0 <= weak_category_probability <= 1:
            raise ValueError(
                "weak_category_probability must be between 0 and 1."
            )

        if not 0 <= random_category_probability <= 1:
            raise ValueError(
                "random_category_probability must be between 0 and 1."
            )

        if abs(
            weak_category_probability
            + random_category_probability
            - 1.0
        ) > 1e-6:
            raise ValueError(
                "Sampling probabilities must sum to 1.0."
            )

        self.weak_category_probability = (
            weak_category_probability
        )

        self.random_category_probability = (
            random_category_probability
        )

        self.min_episodes = min_episodes
        self.rolling_window = rolling_window
        self.use_failure_rate = use_failure_rate

        self.random = random.Random(seed)
        self.logger = logger

        self.category_stats: Dict[str, Dict[str, Any]] = (
            defaultdict(
                lambda: {
                    "episodes": 0,
                    "failures": 0,
                    "failure_rate": 0.0,
                }
            )
        )

        self._scenario_categories = (
            self._build_category_index()
        )

    # ==================================================================
    # SCENARIO INDEX
    # ==================================================================

    def _build_category_index(self) -> Dict[str, List[Any]]:
        """Group scenarios by their attack/security category."""

        categories = defaultdict(list)

        for scenario in self.scenarios:

            category = self._get_category(
                scenario
            )

            categories[category].append(
                scenario
            )

        return dict(categories)

    # ==================================================================
    # PUBLIC SAMPLING
    # ==================================================================

    def sample(
        self,
        count: int = 1,
    ) -> List[Any]:
        """
        Sample scenarios according to the adaptive policy.

        70% of selections target weak categories once enough
        historical evidence exists.

        Otherwise, sampling falls back to random exploration.
        """

        if count <= 0:
            raise ValueError(
                "count must be greater than 0."
            )

        selected = []

        for _ in range(count):

            category = self._select_category()

            if category is None:
                selected.append(
                    self._random_scenario()
                )
                continue

            selected.append(
                self._sample_from_category(
                    category
                )
            )

        return selected

    def sample_one(self) -> Any:
        """Return one scenario."""

        return self.sample(
            count=1
        )[0]

    # ==================================================================
    # CATEGORY SELECTION
    # ==================================================================

    def _select_category(self) -> Optional[str]:
        """
        Choose either a weak category or a random category.

        Weakness-focused sampling is used only when enough
        historical evidence exists.
        """

        available_categories = list(
            self._scenario_categories.keys()
        )

        if not available_categories:
            return None

        weak_categories = (
            self.get_weak_categories()
        )

        use_weakness = (
            bool(weak_categories)
            and self.random.random()
            < self.weak_category_probability
        )

        if use_weakness:
            return self._weighted_weak_category(
                weak_categories
            )

        return self.random.choice(
            available_categories
        )

    def _weighted_weak_category(
        self,
        categories: List[str],
    ) -> str:
        """Sample weak categories proportional to failure rate."""

        if not categories:
            return self.random.choice(
                list(
                    self._scenario_categories.keys()
                )
            )

        weights = []

        for category in categories:

            stats = self.category_stats[
                category
            ]

            if self.use_failure_rate:
                weight = max(
                    float(
                        stats.get(
                            "failure_rate",
                            0.0,
                        )
                    ),
                    1e-6,
                )
            else:
                weight = 1.0

            weights.append(weight)

        return self.random.choices(
            categories,
            weights=weights,
            k=1,
        )[0]

    # ==================================================================
    # SCENARIO SELECTION
    # ==================================================================

    def _sample_from_category(
        self,
        category: str,
    ) -> Any:

        scenarios = (
            self._scenario_categories.get(
                category,
                [],
            )
        )

        if not scenarios:
            return self._random_scenario()

        return self.random.choice(
            scenarios
        )

    def _random_scenario(self) -> Any:

        return self.random.choice(
            self.scenarios
        )

    # ==================================================================
    # METRIC UPDATE
    # ==================================================================

    def update(
        self,
        category: str,
        failed: bool,
    ) -> None:
        """
        Update weakness statistics after an episode.

        The caller decides whether the episode represents a
        failure. This class only records the result.
        """

        category = self._normalize_category(
            category
        )

        stats = self.category_stats[
            category
        ]

        stats["episodes"] += 1

        if failed:
            stats["failures"] += 1

        stats["failure_rate"] = (
            stats["failures"]
            / stats["episodes"]
        )

        self._trim_statistics(
            category
        )

    def update_from_metrics(
        self,
        category: str,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Update statistics from an episode metrics dictionary.

        Supported fields:

            failed
            failure
            success
            reward

        Explicit failure takes priority.
        """

        if "failed" in metrics:
            failed = bool(
                metrics["failed"]
            )

        elif "failure" in metrics:
            failed = bool(
                metrics["failure"]
            )

        elif "success" in metrics:
            failed = not bool(
                metrics["success"]
            )

        elif "reward" in metrics:
            failed = (
                float(
                    metrics["reward"]
                ) < 0
            )

        else:
            raise ValueError(
                "Metrics must contain failed, failure, "
                "success, or reward."
            )

        self.update(
            category=category,
            failed=failed,
        )

    # ==================================================================
    # WEAK CATEGORIES
    # ==================================================================

    def get_weak_categories(
        self,
    ) -> List[str]:
        """
        Return categories with sufficient historical evidence.

        Categories are considered weak when their failure rate
        is greater than or equal to the average observed failure
        rate among sufficiently trained categories.
        """

        eligible = []

        for category, stats in (
            self.category_stats.items()
        ):

            if (
                stats["episodes"]
                < self.min_episodes
            ):
                continue

            eligible.append(
                (
                    category,
                    float(
                        stats["failure_rate"]
                    ),
                )
            )

        if not eligible:
            return []

        average_failure_rate = (
            sum(
                rate
                for _, rate in eligible
            )
            / len(eligible)
        )

        return [
            category
            for category, rate in eligible
            if rate >= average_failure_rate
        ]

    # ==================================================================
    # STATISTICS
    # ==================================================================

    def get_category_stats(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Return a copy of current category statistics."""

        return {
            category: dict(stats)
            for category, stats in (
                self.category_stats.items()
            )
        }

    def get_weakness_report(
        self,
    ) -> Dict[str, Any]:

        weak_categories = (
            self.get_weak_categories()
        )

        return {
            "weak_categories": weak_categories,
            "category_stats": self.get_category_stats(),
            "weak_category_probability": (
                self.weak_category_probability
            ),
            "random_category_probability": (
                self.random_category_probability
            ),
            "min_episodes": self.min_episodes,
            "rolling_window": self.rolling_window,
        }

    # ==================================================================
    # ROLLING WINDOW
    # ==================================================================

    def _trim_statistics(
        self,
        category: str,
    ) -> None:
        """
        Keep statistics bounded.

        For the compact implementation, only aggregate counts
        are maintained. When the rolling window is exceeded,
        counts are proportionally reduced.
        """

        stats = self.category_stats[
            category
        ]

        if (
            self.rolling_window <= 0
            or stats["episodes"]
            <= self.rolling_window
        ):
            return

        ratio = (
            self.rolling_window
            / stats["episodes"]
        )

        stats["episodes"] = max(
            int(
                round(
                    stats["episodes"]
                    * ratio
                )
            ),
            1,
        )

        stats["failures"] = min(
            int(
                round(
                    stats["failures"]
                    * ratio
                )
            ),
            stats["episodes"],
        )

        stats["failure_rate"] = (
            stats["failures"]
            / stats["episodes"]
        )

    # ==================================================================
    # SCENARIO HELPERS
    # ==================================================================

    @staticmethod
    def _get_category(
        scenario: Any,
    ) -> str:

        category = getattr(
            scenario,
            "category",
            None,
        )

        if category is None:
            metadata = getattr(
                scenario,
                "metadata",
                {},
            )

            if isinstance(
                metadata,
                dict,
            ):
                category = metadata.get(
                    "category"
                )

        if category is None:
            return "unknown"

        return WeaknessSampler._normalize_category(
            category
        )

    @staticmethod
    def _normalize_category(
        category: Any,
    ) -> str:

        value = str(
            category
        ).strip()

        return value or "unknown"

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self) -> None:
        """Clear all historical weakness statistics."""

        self.category_stats.clear()

    # ==================================================================
    # PERSISTENCE HELPERS
    # ==================================================================

    def load_report(
        self,
        report: Dict[str, Any],
    ) -> None:
        """Restore category statistics from a saved report."""

        if not isinstance(
            report,
            dict,
        ):
            raise TypeError(
                "Weakness report must be a dictionary."
            )

        stats = report.get(
            "category_stats",
            {},
        )

        if not isinstance(
            stats,
            dict,
        ):
            return

        self.category_stats.clear()

        for category, values in stats.items():

            if not isinstance(
                values,
                dict,
            ):
                continue

            episodes = int(
                values.get(
                    "episodes",
                    0,
                )
            )

            failures = int(
                values.get(
                    "failures",
                    0,
                )
            )

            failures = min(
                max(failures, 0),
                max(episodes, 0),
            )

            self.category_stats[
                category
            ] = {
                "episodes": episodes,
                "failures": failures,
                "failure_rate": (
                    failures / episodes
                    if episodes > 0
                    else 0.0
                ),
            }