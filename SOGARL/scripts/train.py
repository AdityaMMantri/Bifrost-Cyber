"""
train.py

Main SOGARL training entry point.
Flow:

    Dataset
       ↓
    Train/Validation/Test split
       ↓
    Weakness Sampler
       ↓
    Episode Manager
       ↓
    Red + Blue GRPO
       ↓
    Replay Buffer
       ↓
    Metrics
       ↓
    Weakness statistics
       ↓
    Checkpoint
       ↓
    Next episode
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import config

from src.data.scenario_loader import ScenarioLoader
from src.data.metadata_loader import MetadataLoader
from src.data.path_resolver import PathResolver
from src.data.code_loader import CodeLoader
from src.data.models import Scenario

from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator

from src.oracle.oracle import Oracle
from src.oracle.semantic_judge import SemanticJudge
from src.oracle.deterministic_checks import DeterministicChecker
from src.oracle.interaction_checker import InteractionChecker

from src.rl.reward_manager import RewardManager
from src.rl.episode_manager import EpisodeManager
from src.rl.grpo import (
    create_red_trainer,
    create_blue_trainer,
)

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
        help="Disable weakness-based scenario sampling.",
    )

    return parser.parse_args()


# ============================================================================
# LOGGER
# ============================================================================

def build_logger() -> SOGARLLogger:

    log_dir = Path(config.LOG_PATH)
    log_dir.mkdir(parents=True,exist_ok=True)
    return SOGARLLogger(name="SOGARL",log_file=(log_dir/ "training.log"),level=config.LOG_LEVEL,console=True)


# ============================================================================
# DATASET
# ============================================================================

def get_scenario_paths() -> List[Path]:
    """
    Return all scenario directories from the SFT dataset.
    """

    dataset_path = Path(config.DATASET_PATH)

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            "Dataset directory does not exist: "
            f"{dataset_path}")

    scenarios = sorted([path for path in dataset_path.iterdir() if (path.is_dir() and path.name.startswith("scenario_"))],key=lambda path: path.name)

    if not scenarios:
        raise RuntimeError(
            "No scenario directories found in "
            f"{dataset_path}")
    return scenarios


def load_scenarios(scenario_loader: ScenarioLoader,scenario_paths: Sequence[Path]) -> List[Scenario]:
    """
    Load complete Scenario objects.

    The sampler needs Scenario objects because the category is
    stored on the Scenario and is required for weakness-aware
    sampling.
    """

    scenarios: List[Scenario] = []
    for scenario_path in scenario_paths:
        scenario = (scenario_loader.load_scenario(scenario_path.name))
        scenarios.append(scenario)

    if not scenarios:
        raise RuntimeError("No scenarios could be loaded.")

    return scenarios


# ============================================================================
# DATASET SPLIT
# ============================================================================

def split_scenarios(scenarios: Sequence[Scenario]) -> tuple[List[Scenario],List[Scenario],List[Scenario]]:
    """
    Split scenarios into train/validation/test sets.
    The split is deterministic using SPLIT_SEED.
    """
    scenarios = list(scenarios)
    rng = random.Random(config.SPLIT_SEED)
    rng.shuffle(scenarios)
    total = len(scenarios)
    train_count = int(total * config.TRAIN_RATIO)
    validation_count = int(total * config.VALIDATION_RATIO)
    train_end = train_count
    validation_end = (train_count + validation_count)
    train_scenarios = scenarios[:train_end]
    validation_scenarios = scenarios[train_end:validation_end]
    test_scenarios = scenarios[validation_end:]

    if not train_scenarios:
        raise RuntimeError("Training split is empty.")

    return (train_scenarios,validation_scenarios,test_scenarios)


# ============================================================================
# EPISODE MANAGER
# ============================================================================

def build_episode_manager(logger: SOGARLLogger) -> EpisodeManager:
    """
    Construct the complete SOGARL dependency graph.

    Red and Blue use separate role-specific LoRA adapters while
    sharing the underlying base model architecture.

    The SemanticJudge reuses the already-loaded Red generator
    base model and does not load another base-model copy.
    """

    dataset_path = Path(
        config.DATASET_PATH
    )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    metadata_loader = MetadataLoader()

    scenario_loader = ScenarioLoader(
        dataset_path=dataset_path,
        metadata_loader=metadata_loader,
    )

    path_resolver = PathResolver()

    code_loader = CodeLoader(
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    prompt_builder = PromptBuilder()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    red_generator = Generator.from_config(
        "red"
    )

    blue_generator = Generator.from_config(
        "blue"
    )

    # ------------------------------------------------------------------
    # Oracle
    #
    # Ground truth remains Oracle-side.
    # SemanticJudge is therefore constructed here rather than
    # inside PromptBuilder or Generator.
    # ------------------------------------------------------------------

    semantic_judge = SemanticJudge(
        base_generator=red_generator,
        logger=logger,
    )

    oracle = Oracle(
        deterministic_checker=(
            DeterministicChecker()
        ),
        semantic_judge=semantic_judge,
        logger=logger,
    )

    interaction_checker = (
        InteractionChecker()
    )

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    reward_manager = RewardManager(
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Replay + metrics
    # ------------------------------------------------------------------

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
    )

    # ------------------------------------------------------------------
    # GRPO
    # ------------------------------------------------------------------

    red_trainer = create_red_trainer(
        model=red_generator.model,
        logger=logger,
    )

    blue_trainer = create_blue_trainer(
        model=blue_generator.model,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Episode manager
    # ------------------------------------------------------------------

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
    """
    Create the weakness-aware scenario sampler.

    IMPORTANT:
    The sampler receives Scenario objects rather than Path objects
    because it needs the scenario category.
    """

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
# CHECKPOINT MANAGER
# ============================================================================

def build_checkpoint_manager(
    logger: SOGARLLogger,
) -> CheckpointManager:

    checkpoint_root = Path(
        config.CHECKPOINT_PATH
    )

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return CheckpointManager(
        checkpoint_root=checkpoint_root,
        logger=logger,
    )


# ============================================================================
# CHECKPOINT SAVE
# ============================================================================

def save_checkpoint(
    checkpoint_manager: CheckpointManager,
    episode_manager: EpisodeManager,
    sampler: Optional[WeaknessSampler],
    epoch: int,
    episode: int,
    best_metric: Optional[float],
    logger: SOGARLLogger,
) -> Path:

    red_trainer = (
        episode_manager.red_trainer
    )

    blue_trainer = (
        episode_manager.blue_trainer
    )

    if red_trainer is None:

        raise RuntimeError(
            "Red trainer is required for checkpointing."
        )

    if blue_trainer is None:

        raise RuntimeError(
            "Blue trainer is required for checkpointing."
        )

    trainer_state: Dict[str, Any] = {}

    if sampler is not None:

        trainer_state[
            "weakness_sampler"
        ] = sampler.get_weakness_report()

        # WeaknessSampler owns an independent random.Random
        # instance. Persist it so resume does not silently change
        # the future scenario-sampling sequence.
        sampler_rng = getattr(
            sampler,
            "random",
            None,
        )

        if sampler_rng is not None and hasattr(
            sampler_rng,
            "getstate",
        ):

            trainer_state[
                "weakness_sampler_rng_state"
            ] = sampler_rng.getstate()

    checkpoint_name = (
        f"epoch_{epoch:03d}"
        f"_episode_{episode:06d}"
    )

    checkpoint_dir = checkpoint_manager.save(
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
        red_trainer=red_trainer,
        blue_trainer=blue_trainer,
        red_reference_model=(
            red_trainer.get_reference_model()
        ),
        blue_reference_model=(
            blue_trainer.get_reference_model()
        ),
        epoch=epoch,
        episode=episode,
        global_step=episode,
        best_metric=best_metric,
        trainer_state=trainer_state,
    )

    # ReplayBuffer already provides a stable save/load API.
    # Persist it beside the checkpoint so resume does not lose
    # historical interactions.
    replay_buffer = getattr(
        episode_manager,
        "replay_buffer",
        None,
    )

    if replay_buffer is not None:

        replay_save = getattr(
            replay_buffer,
            "save",
            None,
        )

        if replay_save is not None:

            replay_save(
                str(
                    checkpoint_dir
                    / "replay_buffer.json"
                )
            )

    logger.info(
        "Checkpoint saved | "
        f"epoch={epoch} | "
        f"episode={episode}"
    )

    return checkpoint_dir


# ============================================================================
# CHECKPOINT RESTORE
# ============================================================================

def restore_checkpoint(
    checkpoint_manager: CheckpointManager,
    episode_manager: EpisodeManager,
    sampler: Optional[WeaknessSampler],
    logger: SOGARLLogger,
) -> tuple[
    int,
    int,
    Optional[float],
]:
    """
    Restore the latest training checkpoint.

    Returns:

        start_epoch
        global_episode
        best_metric

    The returned epoch is the checkpointed epoch. The training loop
    uses global_episode to skip only the episodes already completed
    in that epoch, allowing correct mid-epoch resume.
    """

    checkpoint = (
        checkpoint_manager.find_latest()
    )

    if checkpoint is None:

        logger.warning(
            "Resume requested but no checkpoint "
            "was found. Starting from scratch."
        )

        return (
            1,
            0,
            None,
        )

    logger.info(
        f"Restoring checkpoint: {checkpoint}"
    )

    red_trainer = (
        episode_manager.red_trainer
    )

    blue_trainer = (
        episode_manager.blue_trainer
    )

    if red_trainer is None:

        raise RuntimeError(
            "Red trainer is required for resume."
        )

    if blue_trainer is None:

        raise RuntimeError(
            "Blue trainer is required for resume."
        )

    state = checkpoint_manager.load(
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
        red_trainer=red_trainer,
        blue_trainer=blue_trainer,
        red_reference_model=(
            red_trainer.get_reference_model()
        ),
        blue_reference_model=(
            blue_trainer.get_reference_model()
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

    episode = int(
        trainer_state.get(
            "episode",
            0,
        )
    )

    best_metric = trainer_state.get(
        "best_metric"
    )

    # --------------------------------------------------------------
    # Restore weakness sampler history.
    #
    # CheckpointManager stores user trainer state under the
    # "trainer_state" key. The previous implementation looked for
    # weakness_sampler at the outer level.
    # --------------------------------------------------------------

    user_trainer_state = trainer_state.get(
        "trainer_state",
        {},
    )

    if not isinstance(
        user_trainer_state,
        dict,
    ):

        user_trainer_state = {}

    if sampler is not None:

        sampler_report = user_trainer_state.get(
            "weakness_sampler"
        )

        if isinstance(
            sampler_report,
            dict,
        ):

            sampler.load_report(
                sampler_report
            )

        sampler_rng_state = (
            user_trainer_state.get(
                "weakness_sampler_rng_state"
            )
        )

        sampler_rng = getattr(
            sampler,
            "random",
            None,
        )

        if (
            sampler_rng is not None
            and sampler_rng_state is not None
            and hasattr(
                sampler_rng,
                "setstate",
            )
        ):

            sampler_rng.setstate(
                sampler_rng_state
            )

    # --------------------------------------------------------------
    # Restore replay buffer.
    # --------------------------------------------------------------

    replay_buffer = getattr(
        episode_manager,
        "replay_buffer",
        None,
    )

    replay_path = (
        checkpoint
        / "replay_buffer.json"
    )

    if (
        replay_buffer is not None
        and replay_path.exists()
    ):

        replay_load = getattr(
            replay_buffer,
            "load",
            None,
        )

        if replay_load is not None:

            replay_load(
                str(replay_path)
            )

    logger.info(
        "Checkpoint restored | "
        f"epoch={epoch} | "
        f"episode={episode}"
    )

    return (
        epoch,
        episode,
        best_metric,
    )


# ============================================================================
# METRICS
# ============================================================================

def aggregate_episode_metrics(
    metrics: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:

    if not metrics:

        return {
            "episodes": 0
        }

    numeric_keys = set()

    for item in metrics:

        for key, value in item.items():

            if isinstance(
                value,
                (int, float),
            ):

                numeric_keys.add(
                    key
                )

    summary: Dict[
        str,
        Any,
    ] = {
        "episodes": len(
            metrics
        )
    }

    for key in sorted(
        numeric_keys
    ):

        values = [
            float(
                item[key]
            )
            for item in metrics
            if (
                key in item
                and isinstance(
                    item[key],
                    (int, float),
                )
            )
        ]

        if values:

            summary[key] = (
                sum(values)
                / len(values)
            )

    return summary


def log_epoch_summary(
    epoch: int,
    metrics: Dict[str, Any],
    logger: SOGARLLogger,
) -> None:

    logger.info(
        f"Epoch {epoch} complete"
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
# WEAKNESS FAILURE SIGNAL
# ============================================================================

def episode_failed(
    result: Any,
) -> bool:
    """
    Convert the completed episode result into the failure signal
    expected by WeaknessSampler.

    Explicit failure/success metrics are preferred.

    For the current SOGARL metrics, the interaction reward is the
    most direct episode-level failure signal because it records the
    Red-versus-Blue outcome after the Top-K interaction.

    Normalized advantages are deliberately never used as a failure
    signal because their group-relative construction centers them
    around zero by design.
    """

    metrics = getattr(
        result,
        "episode_metrics",
        {},
    )

    if not isinstance(
        metrics,
        dict,
    ):

        return False

    if "failed" in metrics:

        return bool(
            metrics["failed"]
        )

    if "failure" in metrics:

        return bool(
            metrics["failure"]
        )

    if "success" in metrics:

        return not bool(
            metrics["success"]
        )

    interaction_reward = metrics.get(
        "blue_mean_interaction_reward"
    )

    if interaction_reward is not None:

        return float(
            interaction_reward
        ) < 0.0

    # No direct episode-level failure signal is available.
    # Do not invent a threshold for the normalized reward.
    return False


# ============================================================================
# SCENARIO SELECTION
# ============================================================================

def select_scenario(
    scenarios: Sequence[Scenario],
    sampler: Optional[WeaknessSampler],
    episode_index: int,
) -> Scenario:

    if sampler is not None:

        return sampler.sample_one()

    return scenarios[
        episode_index
        % len(scenarios)
    ]


# ============================================================================
# VALIDATION
# ============================================================================

def run_validation(
    episode_manager: EpisodeManager,
    validation_scenarios: Sequence[Scenario],
    epoch: int,
    logger: SOGARLLogger,
) -> Dict[str, Any]:
    """
    Evaluate the current policies on the held-out validation split.

    Validation:
        - performs no GRPO update
        - does not update weakness statistics
        - does not update training category history
        - does not use validation scenarios for training selection

    The combined validation reward is the mean of the Red and Blue
    mean Oracle rewards, preserving the two-agent nature of the
    validation metric.
    """

    if not validation_scenarios:

        logger.warning(
            "Validation split is empty; "
            "validation metric cannot be computed."
        )

        return {
            "episodes": 0,
            "combined_validation_reward": None,
        }

    validation_metrics: List[
        Dict[str, Any]
    ] = []

    logger.info(
        f"Starting validation | "
        f"epoch={epoch} | "
        f"scenarios={len(validation_scenarios)}"
    )

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
                    f"epoch_{epoch:03d}_"
                    f"{index:06d}"
                ),
                epoch=epoch,
            )
        )

        metrics = getattr(
            result,
            "episode_metrics",
            {},
        )

        if isinstance(
            metrics,
            dict,
        ) and metrics:

            validation_metrics.append(
                metrics
            )

    summary = aggregate_episode_metrics(
        validation_metrics
    )

    red_reward = summary.get(
        "red_mean_oracle_reward"
    )

    blue_reward = summary.get(
        "blue_mean_oracle_reward"
    )

    if (
        isinstance(
            red_reward,
            (int, float),
        )
        and isinstance(
            blue_reward,
            (int, float),
        )
    ):

        summary[
            "combined_validation_reward"
        ] = (
            float(red_reward)
            + float(blue_reward)
        ) / 2.0

    else:

        summary[
            "combined_validation_reward"
        ] = None

    logger.info(
        f"Validation complete | "
        f"epoch={epoch} | "
        f"metrics={summary}"
    )

    return summary


# ============================================================================
# BEST CHECKPOINT
# ============================================================================

def save_best_checkpoint(
    checkpoint_manager: CheckpointManager,
    source_checkpoint: Path,
    logger: SOGARLLogger,
) -> None:
    """
    Keep a stable 'best' checkpoint copy.

    The CheckpointManager intentionally refuses to overwrite an existing
    checkpoint, so the previous best is removed before copying the new
    best checkpoint directory.
    """

    best_name = "best"

    if checkpoint_manager.exists(
        best_name
    ):

        checkpoint_manager.delete(
            best_name
        )

    best_path = (
        checkpoint_manager.checkpoint_root
        / best_name
    )

    shutil.copytree(
        source_checkpoint,
        best_path,
    )

    logger.info(
        f"Best checkpoint updated: {best_path}"
    )


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

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    config.validate_config()

    config.create_output_directories()

    set_seed(
        config.SEED
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    dataset_path = Path(
        config.DATASET_PATH
    )

    metadata_loader = MetadataLoader()

    scenario_loader = ScenarioLoader(
        dataset_path=dataset_path,
        metadata_loader=metadata_loader,
    )

    scenario_paths = get_scenario_paths()

    scenarios = load_scenarios(
        scenario_loader=scenario_loader,
        scenario_paths=scenario_paths,
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
            scenarios=train_scenarios,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------

    start_epoch = 1

    global_episode = 0

    best_metric = None

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    if resume:

        (
            start_epoch,
            global_episode,
            best_metric,
        ) = restore_checkpoint(
            checkpoint_manager=(
                checkpoint_manager
            ),
            episode_manager=(
                episode_manager
            ),
            sampler=sampler,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

        logger.info(
            "=" * 70
        )

        logger.info(
            f"Starting epoch "
            f"{epoch}/{epochs}"
        )

        logger.info(
            "=" * 70
        )

        epoch_metrics: List[
            Dict[str, Any]
        ] = []

        # --------------------------------------------------------------
        # Number of episodes
        # --------------------------------------------------------------

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

        # On resume, the checkpoint stores the global episode number.
        # Convert that to the number already completed in this epoch.
        # For a normal uninterrupted run this is zero.
        episodes_before_epoch = (
            (epoch - 1)
            * episodes_this_epoch
        )

        if epoch == start_epoch and global_episode > 0:

            resume_completed_in_epoch = max(
                0,
                global_episode
                - episodes_before_epoch,
            )

        else:

            resume_completed_in_epoch = 0

        resume_completed_in_epoch = min(
            resume_completed_in_epoch,
            episodes_this_epoch,
        )

        for episode_index in range(
            resume_completed_in_epoch,
            episodes_this_epoch,
        ):

            global_episode += 1

            scenario = select_scenario(
                scenarios=train_scenarios,
                sampler=sampler,
                episode_index=episode_index,
            )

            logger.info(
                f"Episode {global_episode} | "
                f"Epoch {epoch} | "
                f"Scenario {scenario.scenario_id}"
            )

            # ----------------------------------------------------------
            # Run complete SOGARL episode
            #
            # EpisodeManager handles:
            #
            #   Red G=8
            #   Red Oracle
            #   Red Oracle advantage
            #   Red GRPO
            #   Attack_best / confidence gate
            #   Blue G=8
            #   Blue Oracle
            #   Top-K=3
            #   Turn-3 challenges
            #   interaction rewards
            #   Blue fused advantage
            #   Blue GRPO
            #
            # Exactly one Red update and one Blue update are
            # performed for a training episode.
            # ----------------------------------------------------------

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

            # ----------------------------------------------------------
            # Episode metrics
            # ----------------------------------------------------------

            metrics = getattr(
                result,
                "episode_metrics",
                {},
            )

            if metrics:

                epoch_metrics.append(
                    metrics
                )

            # ----------------------------------------------------------
            # Weakness statistics
            # ----------------------------------------------------------

            if sampler is not None:

                category = getattr(
                    scenario,
                    "category",
                    None,
                )

                if category is None:

                    metadata = getattr(
                        scenario,
                        "metadata",
                        None,
                    )

                    if isinstance(
                        metadata,
                        dict,
                    ):

                        category = metadata.get(
                            "category"
                        )

                if category is None:

                    category = "unknown"

                sampler.update(
                    category=category,
                    failed=episode_failed(
                        result
                    ),
                )

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------

            if (
                config.LOG_EVERY_EPISODES > 0
                and global_episode
                % config.LOG_EVERY_EPISODES
                == 0
            ):

                logger.info(
                    "Episode metrics: "
                    f"{metrics}"
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

                save_checkpoint(
                    checkpoint_manager=(
                        checkpoint_manager
                    ),
                    episode_manager=(
                        episode_manager
                    ),
                    sampler=sampler,
                    epoch=epoch,
                    episode=global_episode,
                    best_metric=best_metric,
                    logger=logger,
                )

        # --------------------------------------------------------------
        # Epoch summary
        # --------------------------------------------------------------

        summary = (
            aggregate_episode_metrics(
                epoch_metrics
            )
        )

        log_epoch_summary(
            epoch=epoch,
            metrics=summary,
            logger=logger,
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        validation_summary = run_validation(
            episode_manager=episode_manager,
            validation_scenarios=validation_scenarios,
            epoch=epoch,
            logger=logger,
        )

        validation_metric = (
            validation_summary.get(
                "combined_validation_reward"
            )
        )

        # --------------------------------------------------------------
        # Best metric / best checkpoint
        # --------------------------------------------------------------

        if isinstance(
            validation_metric,
            (int, float),
        ):

            is_best = (
                best_metric is None
                or float(validation_metric)
                > float(best_metric)
            )

            if is_best:

                best_metric = float(
                    validation_metric
                )

                if (
                    checkpoint_enabled
                    and getattr(
                        config,
                        "SAVE_BEST_CHECKPOINT",
                        True,
                    )
                ):

                    # The epoch checkpoint below is the canonical
                    # checkpoint for this epoch. Save it first, then
                    # make an exact copy as "best".
                    if getattr(
                        config,
                        "SAVE_EVERY_EPOCH",
                        True,
                    ):

                        checkpoint_name = (
                            f"epoch_{epoch:03d}"
                            f"_episode_{global_episode:06d}"
                        )

                        if checkpoint_manager.exists(
                            checkpoint_name
                        ):

                            epoch_checkpoint = (
                                checkpoint_manager
                                .checkpoint_root
                                / checkpoint_name
                            )

                        else:

                            epoch_checkpoint = (
                                save_checkpoint(
                                    checkpoint_manager=(
                                        checkpoint_manager
                                    ),
                                    episode_manager=(
                                        episode_manager
                                    ),
                                    sampler=sampler,
                                    epoch=epoch,
                                    episode=global_episode,
                                    best_metric=best_metric,
                                    logger=logger,
                                )
                            )

                        save_best_checkpoint(
                            checkpoint_manager=(
                                checkpoint_manager
                            ),
                            source_checkpoint=(
                                epoch_checkpoint
                            ),
                            logger=logger,
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

                save_report = getattr(
                    episode_manager
                    .metrics_logger,
                    "save_weakness_report",
                    None,
                )

                if save_report is not None:

                    save_report(
                        report
                    )

        # --------------------------------------------------------------
        # Epoch checkpoint
        #
        # If the best checkpoint was already saved above, do not save
        # the same epoch checkpoint twice.
        # --------------------------------------------------------------

        if (
            checkpoint_enabled
            and getattr(
                config,
                "SAVE_EVERY_EPOCH",
                True,
            )
        ):

            epoch_checkpoint_exists = (
                checkpoint_manager.exists(
                    f"epoch_{epoch:03d}"
                    f"_episode_{global_episode:06d}"
                )
            )

            if not epoch_checkpoint_exists:

                save_checkpoint(
                    checkpoint_manager=(
                        checkpoint_manager
                    ),
                    episode_manager=(
                        episode_manager
                    ),
                    sampler=sampler,
                    epoch=epoch,
                    episode=global_episode,
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

        flush = getattr(
            episode_manager.metrics_logger,
            "flush",
            None,
        )

        if flush is not None:

            flush()

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
                "epochs must be at least 1."
            )

        if (
            max_episodes is not None
            and max_episodes < 1
        ):

            raise ValueError(
                "max_episodes must be at least 1."
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
    sys.exit(main())