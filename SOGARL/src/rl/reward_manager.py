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

    @staticmethod
    def _advantage_epsilon() -> float:
        return float(
            getattr(
                config,
                "ORACLE_ADVANTAGE_EPSILON",
                1e-8,
            )
        )

    @staticmethod
    def _interaction_epsilon() -> float:
        return float(
            getattr(
                config,
                "INTERACTION_STD_EPSILON",
                1e-8,
            )
        )

    @staticmethod
    def _alpha() -> float:
        """
        Locked SOGARL Oracle contribution.

        Default:
            alpha = 0.8

        Beta controls the interaction contribution and is
        scheduled independently.
        """

        return max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        config,
                        "ALPHA",
                        0.8,
                    )
                ),
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

        # SOGARL group-relative Oracle advantage:
        #
        #     A_i = (R_i - mean(R)) / (std(R) + epsilon)
        #
        # The standard deviation is calculated over the current
        # candidate group only.
        std_reward = rewards.std(
            unbiased=False
        )

        advantages = (
            rewards - mean_reward
        ) / (
            std_reward
            + self._advantage_epsilon()
        )

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

        # Only candidates that actually participated in Turn 3
        # are allowed to contribute to the interaction group.
        #
        # Untested candidates are NOT treated as zero-reward
        # observations.
        tested = [
            candidate
            for candidate in candidates
            if candidate.interaction_reward is not None
        ]

        # No Turn-3 interaction was performed.
        #
        # Therefore every candidate receives zero interaction
        # advantage. This means "no evidence", not a reward.
        if not tested:

            for candidate in candidates:
                candidate.interaction_advantage = 0.0

            return candidates

        tested_rewards = torch.tensor(
            [
                float(candidate.interaction_reward)
                for candidate in tested
            ],
            dtype=torch.float32,
        )

        # Interaction advantage is normalized ONLY across
        # candidates actually tested in Turn 3.
        mean_reward = tested_rewards.mean()

        if getattr(
            config,
            "NORMALIZE_INTERACTION_ADVANTAGE",
            True,
        ):

            std_reward = tested_rewards.std(
                unbiased=False
            )

            tested_advantages = (
                tested_rewards - mean_reward
            ) / (
                std_reward
                + self._interaction_epsilon()
            )

        else:

            tested_advantages = (
                tested_rewards - mean_reward
            )

        # Assign normalized interaction advantages to tested
        # candidates only.
        for candidate, advantage in zip(
            tested,
            tested_advantages,
        ):
            candidate.interaction_advantage = float(
                advantage.item()
            )

        # Candidates that were not selected for Top-K / Turn-3
        # testing receive exactly zero interaction advantage.
        for candidate in candidates:
            if candidate.interaction_reward is None:
                candidate.interaction_advantage = 0.0

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

        # Mean Oracle reward for THIS episode.
        #
        # This is deliberately calculated over the complete
        # Blue candidate group.
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

        Then beta increases continuously from 0 to BETA_MAX:

            mean < BETA_START_MEAN
                -> beta = 0

            mean >= BETA_FULL_MEAN
                -> beta = BETA_MAX

            otherwise:
                linear interpolation

        The current episode is intentionally excluded from
        category history when beta is calculated.
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

        # Locked maximum interaction contribution:
        #
        #     beta <= 0.2
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

        # --------------------------------------------------------------
        # 1. Oracle group-relative advantage
        # --------------------------------------------------------------

        self.calculate_oracle_advantages(
            candidates
        )

        # --------------------------------------------------------------
        # 2. Interaction advantage
        #
        # Only tested Top-K candidates participate in the
        # interaction normalization group.
        # --------------------------------------------------------------

        self.calculate_interaction_advantages(
            candidates
        )

        # --------------------------------------------------------------
        # 3. Historical category statistics
        #
        # IMPORTANT:
        # These statistics are read BEFORE the current episode is
        # appended to history.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # 4. Adaptive beta
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # 5. Locked alpha
        #
        #     A_final = alpha * A_oracle
        #              + beta * A_interaction
        #
        # Default alpha = 0.8.
        # Maximum beta = 0.2.
        # --------------------------------------------------------------

        alpha = self._alpha()

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

            # ----------------------------------------------------------
            # Store reward information for replay / metrics / debugging.
            # ----------------------------------------------------------

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

        # Red Turn 1 is trained ONLY from Oracle quality.
        #
        # Turn-3 interaction rewards are NOT used for a second
        # Red GRPO update in SOGARL V1.

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