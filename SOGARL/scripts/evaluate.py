"""
evaluate.py

Evaluate trained SOGARL policies on the held-out test split.

Evaluation:

    Test scenarios
         ↓
    Red Turn 1
         ↓
    Blue Turn 2
         ↓
    Top-K Blue defenses
         ↓
    Red Turn 3
         ↓
    Oracle + interaction evaluation
         ↓
    Metrics
         ↓
    Evaluation report

Evaluation never:
    - updates model weights
    - performs GRPO optimization
    - performs weakness sampling
    - modifies the dataset
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Resolve the project root before importing project-local modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import config

from src.data.scenario_loader import ScenarioLoader
from src.data.metadata_loader import MetadataLoader
from src.data.path_resolver import PathResolver
from src.data.code_loader import CodeLoader

from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator

from src.oracle.oracle import Oracle
from src.oracle.deterministic_checks import DeterministicChecker
from src.oracle.interaction_checker import InteractionChecker
from src.oracle.semantic_judge import SemanticJudge

from src.rl.reward_manager import RewardManager
from src.rl.episode_manager import EpisodeManager

from src.logging.replay_buffer import ReplayBuffer
from src.logging.metrics_logger import MetricsLogger

from src.utils.logger import SOGARLLogger
from src.utils.seed import set_seed
# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SOGARL on the held-out "
            "test split."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint directory. If omitted, "
            "the latest checkpoint is used."
        ),
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help=(
            "Evaluate one scenario from the "
            "test split."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N "
            "test scenarios."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help=(
            "Evaluation report filename."
        ),
    )

    parser.add_argument(
        "--no-replay",
        action="store_true",
        help=(
            "Do not store evaluation episodes "
            "in the replay buffer."
        ),
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
        name="SOGARL-Evaluation",
        log_file=(
            log_dir
            / "evaluation.log"
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
    loader: ScenarioLoader,
) -> List[Path]:

    scenarios = (
        loader.discover_scenarios()
    )

    if not scenarios:

        raise RuntimeError(
            "No scenarios found in: "
            f"{config.DATASET_PATH}"
        )

    return scenarios


def load_scenarios(
    loader: ScenarioLoader,
    paths: List[Path],
) -> List[Any]:

    return [
        loader.load_scenario(
            path.name
        )
        for path in paths
    ]


# ============================================================================
# SPLIT
# ============================================================================

def split_scenarios(
    scenarios: List[Any],
) -> tuple[
    List[Any],
    List[Any],
    List[Any],
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

    validation_start = (
        train_count
    )

    test_start = (
        train_count
        + validation_count
    )

    train_scenarios = scenarios[
        :validation_start
    ]

    validation_scenarios = scenarios[
        validation_start:test_start
    ]

    test_scenarios = scenarios[
        test_start:
    ]

    if not test_scenarios:

        raise RuntimeError(
            "Test split is empty."
        )

    return (
        train_scenarios,
        validation_scenarios,
        test_scenarios,
    )


# ============================================================================
# CHECKPOINT
# ============================================================================

def resolve_checkpoint(
    checkpoint: Optional[str],
    logger: SOGARLLogger,
) -> Path:

    if checkpoint is not None:

        checkpoint_path = Path(
            checkpoint
        )

        if not checkpoint_path.is_absolute():

            checkpoint_path = (
                Path(
                    config.CHECKPOINT_PATH
                )
                / checkpoint_path
            )

    else:

        checkpoint_root = Path(
            config.CHECKPOINT_PATH
        )

        if not checkpoint_root.exists():

            raise FileNotFoundError(
                "Checkpoint directory does not "
                "exist: "
                f"{checkpoint_root}"
            )

        candidates = [
            path
            for path
            in checkpoint_root.iterdir()
            if (
                path.is_dir()
                and (
                    path
                    / "checkpoint_info.json"
                ).exists()
            )
        ]

        if not candidates:

            raise FileNotFoundError(
                "No valid checkpoints found in: "
                f"{checkpoint_root}"
            )

        checkpoint_path = max(
            candidates,
            key=lambda path: (
                _checkpoint_step(path),
                path.stat().st_mtime,
            ),
        )

    if not checkpoint_path.is_dir():

        raise FileNotFoundError(
            "Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    red_adapter = (
        checkpoint_path
        / "red_adapter"
    )

    blue_adapter = (
        checkpoint_path
        / "blue_adapter"
    )

    if not red_adapter.is_dir():

        raise FileNotFoundError(
            "Red adapter missing: "
            f"{red_adapter}"
        )

    if not blue_adapter.is_dir():

        raise FileNotFoundError(
            "Blue adapter missing: "
            f"{blue_adapter}"
        )

    logger.info(
        "Using checkpoint: "
        f"{checkpoint_path}"
    )

    return checkpoint_path


def _checkpoint_step(
    checkpoint_path: Path,
) -> int:

    info_path = (
        checkpoint_path
        / "checkpoint_info.json"
    )

    if not info_path.exists():

        return 0

    try:

        with info_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            info = json.load(
                file
            )

        return int(
            info.get(
                "global_step",
                0,
            )
        )

    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):

        return 0


# ============================================================================
# GENERATORS
# ============================================================================

def build_generators(
    checkpoint_path: Path,
    logger: SOGARLLogger,
) -> tuple[
    Generator,
    Generator,
]:

    red_adapter = (
        checkpoint_path
        / "red_adapter"
    )

    blue_adapter = (
        checkpoint_path
        / "blue_adapter"
    )

    logger.info(
        "Loading trained Red adapter."
    )

    red_generator = (
        Generator.from_adapter(
            role="red",
            adapter_path=red_adapter,
            logger=logger,
            is_trainable=False,
        )
    )

    logger.info(
        "Loading trained Blue adapter."
    )

    blue_generator = (
        Generator.from_adapter(
            role="blue",
            adapter_path=blue_adapter,
            logger=logger,
            is_trainable=False,
        )
    )

    red_generator.eval()
    blue_generator.eval()

    return (
        red_generator,
        blue_generator,
    )


# ============================================================================
# EPISODE MANAGER
# ============================================================================

def build_episode_manager(
    scenario_loader: ScenarioLoader,
    red_generator: Generator,
    blue_generator: Generator,
    logger: SOGARLLogger,
    use_replay: bool,
) -> EpisodeManager:

    metadata_loader = (
        scenario_loader.metadata_loader
    )

    path_resolver = PathResolver(
        logger=logger
    )

    code_loader = CodeLoader(
        logger=logger
    )

    prompt_builder = PromptBuilder()

    deterministic_checker = (
        DeterministicChecker()
    )

    semantic_judge = None

    if config.USE_SEMANTIC_ORACLE:
        semantic_judge = SemanticJudge(
            base_generator=red_generator,
            logger=logger,
        )

    oracle = Oracle(
        deterministic_checker=(
            deterministic_checker
        ),
        semantic_judge=semantic_judge,
        logger=logger,
    )

    interaction_checker = (
        InteractionChecker()
    )

    reward_manager = RewardManager(
        logger=logger
    )

    replay_buffer = None

    if use_replay:

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

    return EpisodeManager(
        scenario_loader=scenario_loader,
        metadata_loader=metadata_loader,
        path_resolver=path_resolver,
        code_loader=code_loader,
        prompt_builder=prompt_builder,
        red_generator=red_generator,
        blue_generator=blue_generator,
        oracle=oracle,
        interaction_checker=(
            interaction_checker
        ),
        reward_manager=reward_manager,
        red_trainer=None,
        blue_trainer=None,
        replay_buffer=replay_buffer,
        metrics_logger=metrics_logger,
        logger=logger,
    )


# ============================================================================
# METRICS
# ============================================================================

def aggregate_metrics(
    metrics_list: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:

    if not metrics_list:

        return {
            "episodes": 0
        }

    numeric_keys = set()

    for metrics in metrics_list:

        for key, value in metrics.items():

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
            metrics_list
        )
    }

    for key in sorted(
        numeric_keys
    ):

        values = [
            float(
                metrics[key]
            )
            for metrics
            in metrics_list
            if (
                key in metrics
                and isinstance(
                    metrics[key],
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


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_scenarios(
    scenarios: List[Any],
    episode_manager: EpisodeManager,
    logger: SOGARLLogger,
) -> Dict[str, Any]:

    results = []

    for index, scenario in enumerate(
        scenarios,
        start=1,
    ):

        logger.info(
            f"Evaluating "
            f"{index}/{len(scenarios)} | "
            f"{scenario.scenario_id}"
        )

        # Evaluation is strictly inference-only:
        # no GRPO updates, no weakness sampling, and no model-weight changes.
        result = (
            episode_manager.run_episode(
                scenario=scenario,
                training=False,
                episode_id=(
                    f"eval_{index:06d}"
                ),
                epoch=0,
            )
        )

        results.append(
            {
                "scenario_id": (
                    scenario.scenario_id
                ),
                "metrics": (
                    result.episode_metrics
                ),
                "result": (
                    result.to_dict()
                ),
            }
        )

    aggregate = aggregate_metrics(
        [
            item["metrics"]
            for item in results
        ]
    )

    return {
        "evaluation": {
            "split": "test",
            "num_scenarios": len(
                scenarios
            ),
        },
        "aggregate_metrics": aggregate,
        "scenarios": results,
    }


# ============================================================================
# REPORT
# ============================================================================

def save_report(
    report: Dict[str, Any],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    args = parse_args()

    logger = build_logger()

    try:

        config.validate_config()

        config.create_output_directories()

        set_seed(
            config.SEED
        )

        # --------------------------------------------------------------
        # Dataset
        # --------------------------------------------------------------

        scenario_loader = (
            build_scenario_loader(
                logger
            )
        )

        scenario_paths = (
            discover_scenarios(
                scenario_loader
            )
        )

        scenarios = load_scenarios(
            scenario_loader,
            scenario_paths,
        )

        (
            _train_scenarios,
            _validation_scenarios,
            test_scenarios,
        ) = split_scenarios(
            scenarios
        )

        # --------------------------------------------------------------
        # Scenario selection
        # --------------------------------------------------------------

        if args.scenario is not None:

            test_scenarios = [
                scenario
                for scenario
                in test_scenarios
                if (
                    scenario.scenario_id
                    == args.scenario
                )
            ]

            if not test_scenarios:

                raise ValueError(
                    f"Scenario '{args.scenario}' "
                    "is not part of the test split."
                )

        # --------------------------------------------------------------
        # Limit
        # --------------------------------------------------------------

        if args.limit is not None:

            if args.limit <= 0:

                raise ValueError(
                    "--limit must be greater than 0."
                )

            test_scenarios = (
                test_scenarios[
                    :args.limit
                ]
            )

        if not test_scenarios:

            raise RuntimeError(
                "No test scenarios selected."
            )

        # --------------------------------------------------------------
        # Checkpoint
        # --------------------------------------------------------------

        checkpoint_path = (
            resolve_checkpoint(
                checkpoint=args.checkpoint,
                logger=logger,
            )
        )

        # --------------------------------------------------------------
        # Models
        # --------------------------------------------------------------

        (
            red_generator,
            blue_generator,
        ) = build_generators(
            checkpoint_path=(
                checkpoint_path
            ),
            logger=logger,
        )

        # --------------------------------------------------------------
        # Episode manager
        # --------------------------------------------------------------

        episode_manager = (
            build_episode_manager(
                scenario_loader=(
                    scenario_loader
                ),
                red_generator=(
                    red_generator
                ),
                blue_generator=(
                    blue_generator
                ),
                logger=logger,
                use_replay=(
                    not args.no_replay
                ),
            )
        )

        # --------------------------------------------------------------
        # Evaluate
        # --------------------------------------------------------------

        report = evaluate_scenarios(
            scenarios=test_scenarios,
            episode_manager=(
                episode_manager
            ),
            logger=logger,
        )

        # --------------------------------------------------------------
        # Report path
        # --------------------------------------------------------------

        output_path = Path(
            args.output
        )

        if not output_path.is_absolute():

            output_path = (
                Path(
                    config.METRICS_PATH
                )
                / output_path
            )

        save_report(
            report=report,
            output_path=output_path,
        )

        # --------------------------------------------------------------
        # Flush metrics
        # --------------------------------------------------------------

        if (
            episode_manager.metrics_logger
            is not None
        ):

            episode_manager.metrics_logger.flush()

        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        aggregate = (
            report[
                "aggregate_metrics"
            ]
        )

        print()
        print("=" * 72)
        print("SOGARL EVALUATION COMPLETE")
        print("=" * 72)

        print(
            f"Checkpoint : "
            f"{checkpoint_path}"
        )

        print(
            f"Scenarios  : "
            f"{len(test_scenarios)}"
        )

        print()

        for key, value in aggregate.items():

            if key == "episodes":

                continue

            if isinstance(
                value,
                float,
            ):

                print(
                    f"{key}: "
                    f"{value:.4f}"
                )

            else:

                print(
                    f"{key}: "
                    f"{value}"
                )

        print()

        print(
            f"Report saved: "
            f"{output_path}"
        )

        print("=" * 72)

        return 0

    except KeyboardInterrupt:

        logger.warning(
            "Evaluation interrupted."
        )

        return 130

    except Exception as exc:

        logger.exception(
            f"Evaluation failed: {exc}"
        )

        return 1


if __name__ == "__main__":

    sys.exit(main())