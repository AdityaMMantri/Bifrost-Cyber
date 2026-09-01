"""
train.py

Main SOGARL RL training entry point.

Pipeline:

    Dataset
       ↓
    Train / Validation / Test split
       ↓
    Scenario objects
       ↓
    Weakness Sampler
       ↓
    Episode Manager
       ↓
    Red GRPO
       ↓
    Blue GRPO
       ↓
    Replay Buffer + Metrics
       ↓
    Validation
       ↓
    Checkpoint
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from configs import config

from src.data.models import Scenario
from src.data.scenario_loader import ScenarioLoader
from src.data.metadata_loader import MetadataLoader
from src.data.path_resolver import PathResolver
from src.data.code_loader import CodeLoader

from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator

from src.oracle.oracle import Oracle
from src.oracle.interaction_checker import InteractionChecker

from src.rl.reward_manager import RewardManager
from src.rl.episode_manager import EpisodeManager

from src.sampling.weakness_sampler import WeaknessSampler

from src.logging.replay_buffer import ReplayBuffer
from src.logging.metrics_logger import MetricsLogger

from src.utils.checkpoint import CheckpointManager
from src.utils.logger import SOGARLLogger
from src.utils.seed import set_seed


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train SOGARL."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override NUM_EPOCHS.",
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum training episodes per epoch.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint.",
    )

    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint saving.",
    )

    parser.add_argument(
        "--no-weakness-sampling",
        action="store_true",
        help="Disable adaptive weakness sampling.",
    )

    return parser.parse_args()


# ============================================================================
# LOGGER
# ============================================================================

def build_logger() -> SOGARLLogger:

    log_dir = Path(
        config.LOG_PATH
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return SOGARLLogger(
        name="SOGARL",
        log_file=(
            log_dir
            / "training.log"
        ),
        level=config.LOG_LEVEL,
        console=True,
    )


# ============================================================================
# DATASET
# ============================================================================

def build_scenario_loader(
    logger: SOGARLLogger,
) -> ScenarioLoader:

    metadata_loader = MetadataLoader(
        logger=logger
    )

    return ScenarioLoader(
        dataset_path=Path(
            config.DATASET_PATH
        ),
        metadata_loader=metadata_loader,
        logger=logger,
        strict=True,
    )


def discover_scenarios(
    scenario_loader: ScenarioLoader,
) -> List[Path]:

    scenarios = (
        scenario_loader
        .discover_scenarios()
    )

    if not scenarios:

        raise RuntimeError(
            "No scenario directories found in: "
            f"{config.DATASET_PATH}"
        )

    return scenarios


def load_scenarios(
    scenario_loader: ScenarioLoader,
    scenario_paths: Sequence[Path],
) -> List[Scenario]:

    scenarios = []

    for path in scenario_paths:

        scenario = (
            scenario_loader
            .load_scenario(
                path.name
            )
        )

        scenarios.append(
            scenario
        )

    return scenarios


# ============================================================================
# DATASET SPLIT
# ============================================================================

def split_scenarios(
    scenarios: Sequence[Scenario],
) -> tuple[
    List[Scenario],
    List[Scenario],
    List[Scenario],
]:

    scenarios = list(
        scenarios
    )

    rng = random.Random(
        config.SPLIT_SEED
    )

    rng.shuffle(
        scenarios
    )

    total = len(
        scenarios
    )

    train_count = int(
        total
        * config.TRAIN_RATIO
    )

    validation_count = int(
        total
        * config.VALIDATION_RATIO
    )

    train_end = train_count

    validation_end = (
        train_end
        + validation_count
    )

    train_scenarios = scenarios[
        :train_end
    ]

    validation_scenarios = scenarios[
        train_end:validation_end
    ]

    test_scenarios = scenarios[
        validation_end:
    ]

    if not train_scenarios:

        raise RuntimeError(
            "Training split is empty."
        )

    return (
        train_scenarios,
        validation_scenarios,
        test_scenarios,
    )


# ============================================================================
# EPISODE MANAGER
# ============================================================================

def build_episode_manager(
    logger: SOGARLLogger,
) -> EpisodeManager:

    dataset_path = Path(
        config.DATASET_PATH
    )

    metadata_loader = MetadataLoader(
        logger=logger
    )

    scenario_loader = ScenarioLoader(
        dataset_path=dataset_path,
        metadata_loader=metadata_loader,
        logger=logger,
        strict=True,
    )

    path_resolver = PathResolver(
        logger=logger
    )

    code_loader = CodeLoader(
        logger=logger
    )

    prompt_builder = PromptBuilder(
        logger=logger
    )

    red_generator = Generator.from_config(
        role="red",
        logger=logger,
    )

    blue_generator = Generator.from_config(
        role="blue",
        logger=logger,
    )

    oracle = Oracle(
        logger=logger
    )

    interaction_checker = (
        InteractionChecker(
            logger=logger
        )
    )

    reward_manager = RewardManager(
        logger=logger
    )

    replay_buffer = ReplayBuffer(
        capacity=getattr(
            config,
            "REPLAY_BUFFER_CAPACITY",
            None,
        ),
        logger=logger,
    )

    metrics_logger = MetricsLogger(
        output_dir=Path(
            config.METRICS_PATH
        ),
        logger=logger,
    )

    red_trainer = None
    blue_trainer = None

    if config.TRAIN_UPDATES_ENABLED:

        from src.rl.grpo import (
            create_red_trainer,
            create_blue_trainer,
        )

        red_trainer = (
            create_red_trainer(
                model=red_generator.model,
                logger=logger,
            )
        )

        blue_trainer = (
            create_blue_trainer(
                model=blue_generator.model,
                logger=logger,
            )
        )

    return EpisodeManager(
        scenario_loader=scenario_loader,
        metadata_loader=metadata_loader,
        path_resolver=path_resolver,
        code_loader=code_loader,
        prompt_builder=prompt_builder,
        red_generator=red_generator,
        blue_generator=blue_generator,
        oracle=oracle,
        interaction_checker=interaction_checker,
        reward_manager=reward_manager,
        red_trainer=red_trainer,
        blue_trainer=blue_trainer,
        replay_buffer=replay_buffer,
        metrics_logger=metrics_logger,
        logger=logger,
    )


# ============================================================================
# WEAKNESS SAMPLER
# ============================================================================

def build_sampler(
    scenarios: Sequence[Scenario],
    logger: SOGARLLogger,
) -> WeaknessSampler:

    return WeaknessSampler(
        scenarios=scenarios,
        weak_category_probability=(
            config.WEAK_CATEGORY_PROBABILITY
        ),
        random_category_probability=(
            config.RANDOM_CATEGORY_PROBABILITY
        ),
        min_episodes=(
            config.WEAKNESS_MIN_EPISODES
        ),
        rolling_window=(
            config.WEAKNESS_ROLLING_WINDOW
        ),
        use_failure_rate=True,
        seed=config.SEED,
        logger=logger,
    )


# ============================================================================
# CHECKPOINT
# ============================================================================

def build_checkpoint_manager(
    logger: SOGARLLogger,
) -> CheckpointManager:

    root = Path(
        config.CHECKPOINT_PATH
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return CheckpointManager(
        checkpoint_root=root,
        logger=logger,
    )


def save_training_checkpoint(
    manager: CheckpointManager,
    episode_manager: EpisodeManager,
    sampler: Optional[WeaknessSampler],
    epoch: int,
    global_episode: int,
    best_metric: Optional[float],
    logger: SOGARLLogger,
) -> Path:

    if (
        episode_manager.red_trainer
        is None
    ):

        raise RuntimeError(
            "Red trainer is required for checkpointing."
        )

    if (
        episode_manager.blue_trainer
        is None
    ):

        raise RuntimeError(
            "Blue trainer is required for checkpointing."
        )

    trainer_state = {}

    if sampler is not None:

        trainer_state[
            "weakness_sampler"
        ] = (
            sampler.get_weakness_report()
        )

    checkpoint_name = (
        f"epoch_{epoch:03d}"
        f"_episode_{global_episode:06d}"
    )

    checkpoint = manager.save(
        checkpoint_name=checkpoint_name,
        red_model=(
            episode_manager
            .red_generator
            .model
        ),
        blue_model=(
            episode_manager
            .blue_generator
            .model
        ),
        red_trainer=(
            episode_manager
            .red_trainer
        ),
        blue_trainer=(
            episode_manager
            .blue_trainer
        ),
        red_reference_model=(
            episode_manager
            .red_trainer
            .get_reference_model()
        ),
        blue_reference_model=(
            episode_manager
            .blue_trainer
            .get_reference_model()
        ),
        epoch=epoch,
        episode=global_episode,
        global_step=(
            global_episode
        ),
        best_metric=best_metric,
        trainer_state=trainer_state,
    )

    logger.info(
        f"Checkpoint saved: {checkpoint}"
    )

    return checkpoint


# ============================================================================
# CHECKPOINT RESTORE
# ============================================================================

def restore_training_checkpoint(
    manager: CheckpointManager,
    episode_manager: EpisodeManager,
    sampler: Optional[WeaknessSampler],
    logger: SOGARLLogger,
) -> tuple[
    int,
    int,
    Optional[float],
]:

    checkpoint = (
        manager.find_latest()
    )

    if checkpoint is None:

        logger.warning(
            "No checkpoint found. "
            "Starting training from scratch."
        )

        return (
            1,
            0,
            None,
        )

    if (
        episode_manager.red_trainer
        is None
        or episode_manager.blue_trainer
        is None
    ):

        raise RuntimeError(
            "Training trainers are required "
            "when resuming."
        )

    state = manager.load(
        checkpoint_path=checkpoint,
        red_model=(
            episode_manager
            .red_generator
            .model
        ),
        blue_model=(
            episode_manager
            .blue_generator
            .model
        ),
        red_trainer=(
            episode_manager
            .red_trainer
        ),
        blue_trainer=(
            episode_manager
            .blue_trainer
        ),
        red_reference_model=(
            episode_manager
            .red_trainer
            .get_reference_model()
        ),
        blue_reference_model=(
            episode_manager
            .blue_trainer
            .get_reference_model()
        ),
        restore_rng=True,
        require_training_state=True,
    )

    trainer_state = state[
        "trainer_state"
    ]

    epoch = int(
        trainer_state.get(
            "epoch",
            0,
        )
    )

    global_episode = int(
        trainer_state.get(
            "episode",
            0,
        )
    )

    best_metric = trainer_state.get(
        "best_metric"
    )

    extra_state = trainer_state.get(
        "trainer_state",
        {}
    )

    if (
        sampler is not None
        and isinstance(
            extra_state,
            dict,
        )
    ):

        sampler_report = (
            extra_state.get(
                "weakness_sampler"
            )
        )

        if isinstance(
            sampler_report,
            dict,
        ):

            sampler.load_report(
                sampler_report
            )

    logger.info(
        "Checkpoint restored | "
        f"epoch={epoch} | "
        f"episode={global_episode}"
    )

    return (
        epoch + 1,
        global_episode,
        best_metric,
    )


# ============================================================================
# METRICS
# ============================================================================

def aggregate_metrics(
    episode_metrics: Sequence[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:

    if not episode_metrics:

        return {
            "episodes": 0
        }

    keys = set()

    for metrics in episode_metrics:

        for key, value in metrics.items():

            if isinstance(
                value,
                (int, float),
            ):

                keys.add(
                    key
                )

    result = {
        "episodes": len(
            episode_metrics
        )
    }

    for key in sorted(
        keys
    ):

        values = [
            float(
                metrics[key]
            )
            for metrics
            in episode_metrics
            if (
                key in metrics
                and isinstance(
                    metrics[key],
                    (int, float),
                )
            )
        ]

        if values:

            result[key] = (
                sum(values)
                / len(values)
            )

    return result


def log_epoch_metrics(
    epoch: int,
    metrics: Dict[str, Any],
    logger: SOGARLLogger,
) -> None:

    logger.info(
        f"Epoch {epoch} summary"
    )

    for key, value in metrics.items():

        if isinstance(
            value,
            float,
        ):

            logger.info(
                f"{key}: {value:.6f}"
            )

        else:

            logger.info(
                f"{key}: {value}"
            )


# ============================================================================
# FAILURE DETECTION
# ============================================================================

def episode_failed(
    result,
) -> bool:
    """
    Determine whether the episode represents
    a Blue failure for weakness sampling.

    Preference order:
        1. Explicit failure metric
        2. Explicit success metric
        3. Mean Blue Oracle reward
    """

    metrics = getattr(
        result,
        "episode_metrics",
        {},
    )

    if "failed" in metrics:

        return bool(
            metrics["failed"]
        )

    if "success" in metrics:

        return not bool(
            metrics["success"]
        )

    reward = metrics.get(
        "blue_mean_oracle_reward"
    )

    if reward is None:

        return False

    return float(
        reward
    ) < 0.5


# ============================================================================
# VALIDATION
# ============================================================================

def run_validation(
    validation_scenarios: Sequence[Scenario],
    episode_manager: EpisodeManager,
    epoch: int,
    logger: SOGARLLogger,
) -> Dict[str, Any]:

    if not validation_scenarios:

        return {
            "validation_episodes": 0,
            "validation_mean_blue_reward": 0.0,
        }

    results = []

    for index, scenario in enumerate(
        validation_scenarios,
        start=1,
    ):

        result = (
            episode_manager.run_episode(
                scenario=scenario,
                training=False,
                episode_id=(
                    f"validation_"
                    f"{epoch:03d}_"
                    f"{index:05d}"
                ),
                epoch=epoch,
            )
        )

        results.append(
            result.episode_metrics
        )

    summary = aggregate_metrics(
        results
    )

    summary[
        "validation_episodes"
    ] = len(
        validation_scenarios
    )

    summary[
        "validation_mean_blue_reward"
    ] = float(
        summary.get(
            "blue_mean_oracle_reward",
            0.0,
        )
    )

    logger.info(
        "Validation complete | "
        f"epoch={epoch} | "
        f"blue_reward="
        f"{summary['validation_mean_blue_reward']:.6f}"
    )

    return summary


# ============================================================================
# TRAINING
# ============================================================================

def train(
    epochs: int,
    max_episodes: Optional[int],
    resume: bool,
    checkpoint_enabled: bool,
    weakness_sampling_enabled: bool,
    logger: SOGARLLogger,
) -> None:

    config.validate_config()

    config.create_output_directories()

    set_seed(
        config.SEED
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    scenario_loader = (
        build_scenario_loader(
            logger
        )
    )

    scenario_paths = discover_scenarios(
        scenario_loader
    )

    scenarios = load_scenarios(
        scenario_loader,
        scenario_paths,
    )

    (
        train_scenarios,
        validation_scenarios,
        test_scenarios,
    ) = split_scenarios(
        scenarios
    )

    logger.info(
        f"Total scenarios: {len(scenarios)}"
    )

    logger.info(
        f"Train scenarios: "
        f"{len(train_scenarios)}"
    )

    logger.info(
        f"Validation scenarios: "
        f"{len(validation_scenarios)}"
    )

    logger.info(
        f"Test scenarios: "
        f"{len(test_scenarios)}"
    )

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    episode_manager = (
        build_episode_manager(
            logger
        )
    )

    checkpoint_manager = (
        build_checkpoint_manager(
            logger
        )
    )

    sampler = None

    if weakness_sampling_enabled:

        sampler = build_sampler(
            train_scenarios,
            logger,
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    start_epoch = 1
    global_episode = 0
    best_metric = None

    if resume:

        (
            start_epoch,
            global_episode,
            best_metric,
        ) = restore_training_checkpoint(
            manager=checkpoint_manager,
            episode_manager=episode_manager,
            sampler=sampler,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Epochs
    # ------------------------------------------------------------------

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

        logger.info(
            "=" * 72
        )

        logger.info(
            f"Epoch {epoch}/{epochs}"
        )

        logger.info(
            "=" * 72
        )

        epoch_metrics = []

        if max_episodes is None:

            episodes_this_epoch = (
                len(train_scenarios)
            )

        else:

            episodes_this_epoch = min(
                max_episodes,
                len(train_scenarios),
            )

        # --------------------------------------------------------------
        # Training episodes
        # --------------------------------------------------------------

        for episode_index in range(
            episodes_this_epoch
        ):

            global_episode += 1

            if sampler is not None:

                scenario = (
                    sampler.sample_one()
                )

            else:

                scenario = (
                    train_scenarios[
                        episode_index
                        % len(train_scenarios)
                    ]
                )

            logger.info(
                f"Episode {global_episode} | "
                f"{scenario.scenario_id}"
            )

            result = (
                episode_manager.run_episode(
                    scenario=scenario,
                    training=True,
                    episode_id=(
                        f"episode_"
                        f"{global_episode:06d}"
                    ),
                    epoch=epoch,
                )
            )

            metrics = (
                result.episode_metrics
            )

            if metrics:

                epoch_metrics.append(
                    metrics
                )

            # ----------------------------------------------------------
            # Weakness statistics
            # ----------------------------------------------------------

            if sampler is not None:

                sampler.update(
                    category=(
                        getattr(
                            scenario,
                            "category",
                            None,
                        )
                        or getattr(
                            scenario,
                            "metadata",
                            {},
                        ).get(
                            "category",
                            "unknown",
                        )
                    ),
                    failed=episode_failed(
                        result
                    ),
                )

            # ----------------------------------------------------------
            # Periodic logging
            # ----------------------------------------------------------

            if (
                config.LOG_EVERY_EPISODES > 0
                and global_episode
                % config.LOG_EVERY_EPISODES
                == 0
            ):

                logger.info(
                    f"Episode metrics: {metrics}"
                )

            # ----------------------------------------------------------
            # Periodic checkpoint
            # ----------------------------------------------------------

            if (
                checkpoint_enabled
                and config.SAVE_EVERY_N_EPISODES > 0
                and global_episode
                % config.SAVE_EVERY_N_EPISODES
                == 0
            ):

                save_training_checkpoint(
                    manager=checkpoint_manager,
                    episode_manager=episode_manager,
                    sampler=sampler,
                    epoch=epoch,
                    global_episode=global_episode,
                    best_metric=best_metric,
                    logger=logger,
                )

        # --------------------------------------------------------------
        # Epoch metrics
        # --------------------------------------------------------------

        epoch_summary = aggregate_metrics(
            epoch_metrics
        )

        log_epoch_metrics(
            epoch=epoch,
            metrics=epoch_summary,
            logger=logger,
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        validation_summary = (
            run_validation(
                validation_scenarios=(
                    validation_scenarios
                ),
                episode_manager=(
                    episode_manager
                ),
                epoch=epoch,
                logger=logger,
            )
        )

        validation_metric = float(
            validation_summary.get(
                "validation_mean_blue_reward",
                0.0,
            )
        )

        # --------------------------------------------------------------
        # Best checkpoint metric
        # --------------------------------------------------------------

        improved = (
            best_metric is None
            or validation_metric
            > best_metric
        )

        if improved:

            best_metric = validation_metric

            logger.info(
                "New best validation metric: "
                f"{best_metric:.6f}"
            )

        # --------------------------------------------------------------
        # Save weakness report
        # --------------------------------------------------------------

        if sampler is not None:

            report = (
                sampler.get_weakness_report()
            )

            if (
                episode_manager.metrics_logger
                is not None
            ):

                episode_manager.metrics_logger \
                    .save_weakness_report(
                        report
                    )

        # --------------------------------------------------------------
        # Epoch checkpoint
        # --------------------------------------------------------------

        if (
            checkpoint_enabled
            and config.SAVE_EVERY_EPOCH
        ):

            save_training_checkpoint(
                manager=checkpoint_manager,
                episode_manager=episode_manager,
                sampler=sampler,
                epoch=epoch,
                global_episode=global_episode,
                best_metric=best_metric,
                logger=logger,
            )

    # ------------------------------------------------------------------
    # Flush metrics
    # ------------------------------------------------------------------

    if (
        episode_manager.metrics_logger
        is not None
    ):

        episode_manager.metrics_logger.flush()

    logger.info(
        "SOGARL training completed."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    args = parse_args()

    logger = build_logger()

    try:

        epochs = (
            args.epochs
            if args.epochs is not None
            else config.NUM_EPOCHS
        )

        max_episodes = (
            args.max_episodes
            if args.max_episodes is not None
            else getattr(
                config,
                "MAX_EPISODES",
                None,
            )
        )

        if epochs < 1:

            raise ValueError(
                "epochs must be >= 1."
            )

        if (
            max_episodes is not None
            and max_episodes < 1
        ):

            raise ValueError(
                "max_episodes must be >= 1."
            )

        train(
            epochs=epochs,
            max_episodes=max_episodes,
            resume=args.resume,
            checkpoint_enabled=(
                not args.no_checkpoint
            ),
            weakness_sampling_enabled=(
                not args.no_weakness_sampling
            ),
            logger=logger,
        )

        return 0

    except KeyboardInterrupt:

        logger.warning(
            "Training interrupted."
        )

        return 130

    except Exception as exc:

        logger.exception(
            f"Training failed: {exc}"
        )

        return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )