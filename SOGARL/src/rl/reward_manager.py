"""
reward_manager.py

Converts Oracle and interaction rewards into GRPO advantages.

Responsibilities:
    - calculate Oracle advantages
    - calculate interaction advantages
    - maintain category-level reward history
    - calculate category-relative beta
    - fuse Oracle and interaction advantages
    - provide reward statistics

Does NOT:
    - generate responses
    - evaluate vulnerabilities
    - decide interaction cases
    - perform GRPO updates
    - select scenarios
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

import torch

from configs import config
from src.data.models import ScoredCandidate


class RewardManager:

    def __init__(
        self,
        logger=None,
    ) -> None:

        self.logger = logger

        # Historical Oracle reward per category.
        #
        # Example:
        #
        # IDOR:
        #     [0.71, 0.83, 0.76, ...]
        #
        # SQL Injection:
        #     [0.61, 0.74, 0.69, ...]
        #
        # Used for category-relative beta scheduling.
        self._category_rewards: Dict[
            str,
            Deque[float],
        ] = defaultdict(
            lambda: deque(
                maxlen=self._history_size()
            )
        )

    # ==================================================================
    # CONFIG HELPERS
    # ==================================================================

    @staticmethod
    def _history_size() -> int:

        return max(
            1,
            int(
                getattr(
                    config,
                    "CATEGORY_REWARD_HISTORY_SIZE",
                    100,
                )
            ),
        )

    @staticmethod
    def _minimum_category_episodes() -> int:

        return max(
            1,
            int(
                getattr(
                    config,
                    "MIN_CATEGORY_EPISODES_FOR_GATE",
                    20,
                )
            ),
        )

    # ==================================================================
    # ORACLE ADVANTAGES
    # ==================================================================

    def calculate_oracle_advantages(
        self,
        candidates: List[ScoredCandidate],
    ) -> List[ScoredCandidate]:

        if not candidates:
            return candidates

        rewards = torch.tensor(
            [
                float(candidate.oracle_reward)
                for candidate in candidates
            ],
            dtype=torch.float32,
        )

        mean_reward = rewards.mean()

        advantages = rewards - mean_reward

        for candidate, advantage in zip(
            candidates,
            advantages,
        ):
            candidate.oracle_advantage = float(
                advantage.item()
            )

        return candidates

    # ==================================================================
    # INTERACTION ADVANTAGES
    # ==================================================================

    def calculate_interaction_advantages(
        self,
        candidates: List[ScoredCandidate],
    ) -> List[ScoredCandidate]:

        if not candidates:
            return candidates

        rewards = torch.tensor(
            [
                (
                    float(candidate.interaction_reward)
                    if candidate.interaction_reward is not None
                    else 0.0
                )
                for candidate in candidates
            ],
            dtype=torch.float32,
        )

        tested = [
            candidate
            for candidate in candidates
            if candidate.interaction_reward is not None
        ]

        if not tested:

            for candidate in candidates:
                candidate.interaction_advantage = 0.0

            return candidates

        # Missing interaction reward is treated as zero evidence.
        #
        # This is intentional.
        #
        # If only one candidate is challenged:
        #
        #     interaction reward = +1
        #
        # it must still produce a positive training signal.
        #
        # Centering only over tested candidates would produce:
        #
        #     +1 - +1 = 0
        #
        # and completely remove the interaction signal.
        mean_reward = rewards.mean()

        advantages = rewards - mean_reward

        if getattr(
            config,
            "NORMALIZE_INTERACTION_ADVANTAGE",
            True,
        ):

            std = rewards.std(
                unbiased=False
            )

            advantages = (
                advantages
                / (
                    std
                    + getattr(
                        config,
                        "INTERACTION_STD_EPSILON",
                        1e-8,
                    )
                )
            )

        for candidate, advantage in zip(
            candidates,
            advantages,
        ):

            if candidate.interaction_reward is None:
                candidate.interaction_advantage = 0.0
            else:
                candidate.interaction_advantage = float(
                    advantage.item()
                )

        return candidates

    # ==================================================================
    # CATEGORY HISTORY
    # ==================================================================

    def update_category_statistics(
        self,
        category: str,
        candidates: List[ScoredCandidate],
    ) -> Dict[str, float]:

        if not category:
            category = "unknown"

        if not candidates:
            return self.get_category_statistics(
                category
            )

        mean_reward = sum(
            float(candidate.oracle_reward)
            for candidate in candidates
        ) / len(candidates)

        history = self._category_rewards[
            category
        ]

        history.append(mean_reward)

        statistics = self.get_category_statistics(
            category
        )

        if self.logger:
            self.logger.info(
                f"Category '{category}' reward history: "
                f"episodes={statistics['episode_count']}, "
                f"mean={statistics['mean_reward']:.4f}"
            )

        return statistics

    def get_category_statistics(
        self,
        category: str,
    ) -> Dict[str, float]:

        if not category:
            category = "unknown"

        history = self._category_rewards.get(
            category
        )

        if not history:
            return {
                "episode_count": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
            }

        values = torch.tensor(
            list(history),
            dtype=torch.float32,
        )

        return {
            "episode_count": int(
                values.numel()
            ),
            "mean_reward": float(
                values.mean().item()
            ),
            "std_reward": float(
                values.std(
                    unbiased=False
                ).item()
            ),
        }

    # ==================================================================
    # BETA SCHEDULE
    # ==================================================================

    def calculate_beta(
        self,
        mean_oracle_reward: float,
        category_episode_count: Optional[int] = None,
    ) -> float:
        """
        Calculate interaction-reward weight.

        Beta is disabled until enough historical category
        episodes exist.

        Then:

            mean < BETA_START_MEAN
                -> beta = 0

            mean >= BETA_FULL_MEAN
                -> beta = BETA_MAX

            otherwise:
                linear interpolation
        """

        if not getattr(
            config,
            "USE_CONTINUOUS_BETA",
            True,
        ):
            return float(
                getattr(
                    config,
                    "BETA_MAX",
                    0.2,
                )
            )

        minimum_episodes = (
            self._minimum_category_episodes()
        )

        if (
            category_episode_count is None
            or category_episode_count
            < minimum_episodes
        ):
            return 0.0

        start_mean = float(
            getattr(
                config,
                "BETA_START_MEAN",
                0.5,
            )
        )

        full_mean = float(
            getattr(
                config,
                "BETA_FULL_MEAN",
                0.8,
            )
        )

        max_beta = float(
            getattr(
                config,
                "BETA_MAX",
                0.2,
            )
        )

        if full_mean <= start_mean:
            raise ValueError(
                "BETA_FULL_MEAN must be greater than "
                "BETA_START_MEAN."
            )

        if mean_oracle_reward < start_mean:
            return 0.0

        if mean_oracle_reward >= full_mean:
            return max_beta

        progress = (
            mean_oracle_reward - start_mean
        ) / (
            full_mean - start_mean
        )

        return max(
            0.0,
            min(
                max_beta,
                max_beta * progress,
            ),
        )

    # ==================================================================
    # FINAL ADVANTAGES
    # ==================================================================

    def calculate_final_advantages(
        self,
        candidates: List[ScoredCandidate],
        category: Optional[str] = None,
    ) -> List[ScoredCandidate]:

        if not candidates:
            return candidates

        self.calculate_oracle_advantages(
            candidates
        )

        self.calculate_interaction_advantages(
            candidates
        )

        category_statistics = (
            self.get_category_statistics(
                category
            )
            if category
            else {
                "episode_count": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
            }
        )

        beta = self.calculate_beta(
            mean_oracle_reward=(
                category_statistics[
                    "mean_reward"
                ]
            ),
            category_episode_count=(
                category_statistics[
                    "episode_count"
                ]
            ),
        )

        alpha = min(
            1.0,
            float(
                getattr(
                    config,
                    "ALPHA_MAX",
                    1.0,
                )
            ),
        )

        for candidate in candidates:

            oracle_advantage = (
                candidate.oracle_advantage
                if candidate.oracle_advantage
                is not None
                else 0.0
            )

            interaction_advantage = (
                candidate.interaction_advantage
                if candidate.interaction_advantage
                is not None
                else 0.0
            )

            candidate.final_advantage = (
                alpha * oracle_advantage
                +
                beta * interaction_advantage
            )

            candidate.metadata[
                "reward_alpha"
            ] = alpha

            candidate.metadata[
                "reward_beta"
            ] = beta

            candidate.metadata[
                "category_episode_count"
            ] = category_statistics[
                "episode_count"
            ]

            candidate.metadata[
                "category_mean_oracle_reward"
            ] = category_statistics[
                "mean_reward"
            ]

        return candidates

    # ==================================================================
    # RED ADVANTAGES
    # ==================================================================

    def calculate_red_advantages(
        self,
        candidates: List[ScoredCandidate],
        category: Optional[str] = None,
    ) -> List[ScoredCandidate]:

        # Red Turn 1 is trained purely from Oracle quality.
        self.calculate_oracle_advantages(
            candidates
        )

        for candidate in candidates:
            candidate.final_advantage = (
                candidate.oracle_advantage
                if candidate.oracle_advantage
                is not None
                else 0.0
            )

            candidate.metadata[
                "reward_alpha"
            ] = 1.0

            candidate.metadata[
                "reward_beta"
            ] = 0.0

        return candidates

    # ==================================================================
    # COMPLETE BLUE PIPELINE
    # ==================================================================

    def process_blue_candidates(
        self,
        candidates: List[ScoredCandidate],
        category: str,
    ) -> List[ScoredCandidate]:
        """
        Calculate Blue advantages using historical category
        statistics, then update the category history.

        The ordering is intentional:

            1. Read historical statistics
            2. Calculate current advantages
            3. Update history

        Therefore the current episode does NOT influence its
        own beta calculation.
        """

        self.calculate_final_advantages(
            candidates=candidates,
            category=category,
        )

        self.update_category_statistics(
            category=category,
            candidates=candidates,
        )

        return candidates

    # ==================================================================
    # REWARD SUMMARY
    # ==================================================================

    def summarize(
        self,
        candidates: List[ScoredCandidate],
    ) -> Dict[str, Any]:

        if not candidates:
            return {
                "count": 0,
                "mean_oracle_reward": 0.0,
                "mean_oracle_advantage": 0.0,
                "mean_interaction_reward": 0.0,
                "mean_interaction_advantage": 0.0,
                "mean_final_advantage": 0.0,
                "tested_candidates": 0,
            }

        oracle_rewards = [
            float(candidate.oracle_reward)
            for candidate in candidates
        ]

        oracle_advantages = [
            float(candidate.oracle_advantage)
            for candidate in candidates
            if candidate.oracle_advantage
            is not None
        ]

        interaction_rewards = [
            float(candidate.interaction_reward)
            for candidate in candidates
            if candidate.interaction_reward
            is not None
        ]

        interaction_advantages = [
            float(candidate.interaction_advantage)
            for candidate in candidates
            if candidate.interaction_advantage
            is not None
        ]

        final_advantages = [
            float(candidate.final_advantage)
            for candidate in candidates
            if candidate.final_advantage
            is not None
        ]

        return {
            "count": len(candidates),

            "mean_oracle_reward": (
                sum(oracle_rewards)
                / len(oracle_rewards)
            ),

            "mean_oracle_advantage": (
                sum(oracle_advantages)
                / len(oracle_advantages)
                if oracle_advantages
                else 0.0
            ),

            "mean_interaction_reward": (
                sum(interaction_rewards)
                / len(interaction_rewards)
                if interaction_rewards
                else 0.0
            ),

            "mean_interaction_advantage": (
                sum(interaction_advantages)
                / len(interaction_advantages)
                if interaction_advantages
                else 0.0
            ),

            "mean_final_advantage": (
                sum(final_advantages)
                / len(final_advantages)
                if final_advantages
                else 0.0
            ),

            "tested_candidates": len(
                interaction_rewards
            ),
        }

    # ==================================================================
    # DEBUG
    # ==================================================================

    def explain_candidate(
        self,
        candidate: ScoredCandidate,
    ) -> Dict[str, Any]:

        return {
            "candidate_index": (
                candidate.candidate_index
            ),

            "oracle_reward": (
                candidate.oracle_reward
            ),

            "oracle_advantage": (
                candidate.oracle_advantage
            ),

            "interaction_reward": (
                candidate.interaction_reward
            ),

            "interaction_advantage": (
                candidate.interaction_advantage
            ),

            "final_advantage": (
                candidate.final_advantage
            ),

            "alpha": candidate.metadata.get(
                "reward_alpha"
            ),

            "beta": candidate.metadata.get(
                "reward_beta"
            ),

            "category_episode_count": (
                candidate.metadata.get(
                    "category_episode_count"
                )
            ),

            "category_mean_oracle_reward": (
                candidate.metadata.get(
                    "category_mean_oracle_reward"
                )
            ),
        }