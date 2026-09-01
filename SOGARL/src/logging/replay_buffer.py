"""
replay_buffer.py

Stores SOGARL episode records for replay, debugging,
evaluation, and training analysis.

This module does not:
    - calculate rewards
    - calculate advantages
    - update models
    - perform GRPO
    - evaluate attacks or defenses
"""

from __future__ import annotations

import json
import random

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# REPLAY RECORD
# ============================================================================

@dataclass
class ReplayRecord:

    scenario_id: str

    episode_id: str

    epoch: int = 0

    global_step: int = 0

    category: str = "unknown"

    red_attack_candidates: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    blue_defense_candidates: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    top_k_blue_candidates: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    interaction_results: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    episode_metrics: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    metadata: Dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    # ==================================================================
    # SERIALIZATION
    # ==================================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "scenario_id": self.scenario_id,
            "episode_id": self.episode_id,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "category": self.category,
            "red_attack_candidates": (
                self.red_attack_candidates
            ),
            "blue_defense_candidates": (
                self.blue_defense_candidates
            ),
            "top_k_blue_candidates": (
                self.top_k_blue_candidates
            ),
            "interaction_results": (
                self.interaction_results
            ),
            "episode_metrics": (
                self.episode_metrics
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "ReplayRecord":

        if not isinstance(
            data,
            dict,
        ):

            raise TypeError(
                "Replay record must be a dictionary."
            )

        required = {
            "scenario_id",
            "episode_id",
        }

        missing = (
            required
            - data.keys()
        )

        if missing:

            raise ValueError(
                "Missing replay record fields: "
                + ", ".join(
                    sorted(missing)
                )
            )

        return cls(
            scenario_id=str(
                data["scenario_id"]
            ),
            episode_id=str(
                data["episode_id"]
            ),
            epoch=int(
                data.get(
                    "epoch",
                    0,
                )
            ),
            global_step=int(
                data.get(
                    "global_step",
                    0,
                )
            ),
            category=str(
                data.get(
                    "category",
                    "unknown",
                )
            ),
            red_attack_candidates=(
                data.get(
                    "red_attack_candidates",
                    [],
                )
            ),
            blue_defense_candidates=(
                data.get(
                    "blue_defense_candidates",
                    [],
                )
            ),
            top_k_blue_candidates=(
                data.get(
                    "top_k_blue_candidates",
                    [],
                )
            ),
            interaction_results=(
                data.get(
                    "interaction_results",
                    [],
                )
            ),
            episode_metrics=(
                data.get(
                    "episode_metrics",
                    {},
                )
            ),
            metadata=(
                data.get(
                    "metadata",
                    {},
                )
            ),
        )


# ============================================================================
# REPLAY BUFFER
# ============================================================================

class ReplayBuffer:

    def __init__(
        self,
        capacity: Optional[int] = None,
        logger=None,
    ) -> None:

        if (
            capacity is not None
            and capacity <= 0
        ):

            raise ValueError(
                "capacity must be greater than 0."
            )

        self.capacity = capacity

        self.logger = logger

        self._records: List[
            ReplayRecord
        ] = []

    # ==================================================================
    # ADD
    # ==================================================================

    def add(
        self,
        record: ReplayRecord,
    ) -> None:

        if not isinstance(
            record,
            ReplayRecord,
        ):

            raise TypeError(
                "record must be a ReplayRecord."
            )

        self._records.append(
            record
        )

        if (
            self.capacity is not None
            and len(self._records)
            > self.capacity
        ):

            self._records.pop(
                0
            )

        if self.logger:

            self.logger.info(
                "Replay record added | "
                f"episode={record.episode_id} | "
                f"size={len(self._records)}"
            )

    def add_dict(
        self,
        record: Dict[str, Any],
    ) -> None:

        self.add(
            ReplayRecord.from_dict(
                record
            )
        )

    # ==================================================================
    # ACCESS
    # ==================================================================

    def get(
        self,
        index: int,
    ) -> ReplayRecord:

        if not isinstance(
            index,
            int,
        ):

            raise TypeError(
                "index must be an integer."
            )

        return self._records[
            index
        ]

    def latest(
        self,
    ) -> Optional[
        ReplayRecord
    ]:

        if not self._records:

            return None

        return self._records[-1]

    def all(
        self,
    ) -> List[
        ReplayRecord
    ]:

        return list(
            self._records
        )

    def sample(
        self,
        size: int,
    ) -> List[
        ReplayRecord
    ]:

        if size <= 0:

            raise ValueError(
                "size must be greater than 0."
            )

        if size > len(
            self._records
        ):

            raise ValueError(
                f"Requested {size} records, "
                f"but buffer contains "
                f"{len(self._records)}."
            )

        return random.sample(
            self._records,
            size,
        )

    # ==================================================================
    # FILTERING
    # ==================================================================

    def by_category(
        self,
        category: str,
    ) -> List[
        ReplayRecord
    ]:

        category = str(
            category
        )

        return [
            record
            for record
            in self._records
            if record.category
            == category
        ]

    def by_scenario(
        self,
        scenario_id: str,
    ) -> List[
        ReplayRecord
    ]:

        return [
            record
            for record
            in self._records
            if record.scenario_id
            == scenario_id
        ]

    def by_episode(
        self,
        episode_id: str,
    ) -> Optional[
        ReplayRecord
    ]:

        for record in self._records:

            if (
                record.episode_id
                == episode_id
            ):

                return record

        return None

    # ==================================================================
    # REWARD ANALYSIS
    # ==================================================================

    def reward_summary(
        self,
    ) -> Dict[str, float]:

        red_rewards: List[
            float
        ] = []

        blue_rewards: List[
            float
        ] = []

        interaction_rewards: List[
            float
        ] = []

        final_advantages: List[
            float
        ] = []

        for record in self._records:

            for candidate in (
                record.red_attack_candidates
            ):

                reward = candidate.get(
                    "oracle_reward"
                )

                if reward is not None:

                    red_rewards.append(
                        float(reward)
                    )

            for candidate in (
                record.blue_defense_candidates
            ):

                reward = candidate.get(
                    "oracle_reward"
                )

                if reward is not None:

                    blue_rewards.append(
                        float(reward)
                    )

                interaction = candidate.get(
                    "interaction_reward"
                )

                if interaction is not None:

                    interaction_rewards.append(
                        float(interaction)
                    )

                final_advantage = (
                    candidate.get(
                        "final_advantage"
                    )
                )

                if final_advantage is not None:

                    final_advantages.append(
                        float(
                            final_advantage
                        )
                    )

        return {
            "red_mean_oracle_reward": (
                self._mean(
                    red_rewards
                )
            ),
            "blue_mean_oracle_reward": (
                self._mean(
                    blue_rewards
                )
            ),
            "blue_mean_interaction_reward": (
                self._mean(
                    interaction_rewards
                )
            ),
            "blue_mean_final_advantage": (
                self._mean(
                    final_advantages
                )
            ),
        }

    # ==================================================================
    # MANAGEMENT
    # ==================================================================

    def clear(
        self,
    ) -> None:

        self._records.clear()

    def __len__(
        self,
    ) -> int:

        return len(
            self._records
        )

    def is_empty(
        self,
    ) -> bool:

        return len(
            self._records
        ) == 0

    def is_full(
        self,
    ) -> bool:

        if self.capacity is None:

            return False

        return len(
            self._records
        ) >= self.capacity

    # ==================================================================
    # SERIALIZATION
    # ==================================================================

    def to_list(
        self,
    ) -> List[
        Dict[str, Any]
    ]:

        return [
            record.to_dict()
            for record
            in self._records
        ]

    def save(
        self,
        path: str | Path,
    ) -> None:

        output_path = Path(
            path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "capacity": self.capacity,
            "records": self.to_list(),
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

        if self.logger:

            self.logger.info(
                "Replay buffer saved: "
                f"{output_path}"
            )

    def load(
        self,
        path: str | Path,
    ) -> None:

        input_path = Path(
            path
        )

        if not input_path.exists():

            raise FileNotFoundError(
                f"Replay file not found: "
                f"{input_path}"
            )

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        # --------------------------------------------------------------
        # Support both the new wrapped format and the old raw-list
        # format.
        # --------------------------------------------------------------

        if isinstance(
            data,
            dict,
        ):

            records = data.get(
                "records",
                [],
            )

        elif isinstance(
            data,
            list,
        ):

            records = data

        else:

            raise ValueError(
                "Replay file must contain "
                "a record list or replay object."
            )

        if not isinstance(
            records,
            list,
        ):

            raise ValueError(
                "Replay records must be a list."
            )

        self.clear()

        for item in records:

            self.add_dict(
                item
            )

        if self.logger:

            self.logger.info(
                "Replay buffer loaded: "
                f"{input_path} | "
                f"records={len(self._records)}"
            )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    def summary(
        self,
    ) -> Dict[str, Any]:

        categories: Dict[
            str,
            int,
        ] = {}

        for record in self._records:

            categories[
                record.category
            ] = (
                categories.get(
                    record.category,
                    0,
                )
                + 1
            )

        return {
            "size": len(
                self._records
            ),
            "capacity": self.capacity,
            "is_empty": self.is_empty(),
            "is_full": self.is_full(),
            "categories": categories,
            "reward_summary": (
                self.reward_summary()
            ),
        }

    # ==================================================================
    # INTERNAL
    # ==================================================================

    @staticmethod
    def _mean(
        values: List[float],
    ) -> float:

        if not values:

            return 0.0

        return float(
            sum(values)
            / len(values)
        )


# ============================================================================
# FACTORY
# ============================================================================

def create_replay_buffer(
    capacity: Optional[int] = None,
    logger=None,
) -> ReplayBuffer:

    return ReplayBuffer(
        capacity=capacity,
        logger=logger,
    )