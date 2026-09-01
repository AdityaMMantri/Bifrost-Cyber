"""
test_episode.py

Run one complete SOGARL episode as an integration test.

Default mode:
    - one scenario
    - Red/Blue generation enabled
    - Oracle enabled
    - interaction evaluation enabled
    - reward calculation enabled
    - GRPO updates disabled

This verifies that the current SOGARL components are compatible
without changing model weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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

from src.rl.reward_manager import RewardManager
from src.rl.episode_manager import EpisodeManager

from src.logging.replay_buffer import ReplayBuffer
from src.logging.metrics_logger import MetricsLogger


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run one complete SOGARL "
            "episode integration test."
        )
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="scenario_001",
        help=(
            "Scenario to test. "
            "Default: scenario_001."
        ),
    )

    parser.add_argument(
        "--update",
        action="store_true",
        help=(
            "Enable GRPO updates. "
            "Do not use for the normal integration test."
        ),
    )

    return parser.parse_args()


# ============================================================================
# SCENARIO
# ============================================================================

def build_scenario_loader() -> ScenarioLoader:

    dataset_path = Path(
        config.DATASET_PATH
    )

    metadata_loader = MetadataLoader()

    return ScenarioLoader(
        dataset_path=dataset_path,
        metadata_loader=metadata_loader,
        strict=True,
    )


def load_scenario(
    scenario_id: str,
) -> Any:

    loader = build_scenario_loader()

    scenario = (
        loader.load_scenario(
            scenario_id
        )
    )

    return scenario


# ============================================================================
# GENERATORS
# ============================================================================

def build_generators():
    """
    Load the Red and Blue SFT policies through Generator.

    Generator owns:
        - model
        - tokenizer
        - device
        - generation
        - log-probability calculation
    """

    red_generator = (
        Generator.from_config(
            role="red"
        )
    )

    blue_generator = (
        Generator.from_config(
            role="blue"
        )
    )

    red_generator.model.eval()
    blue_generator.model.eval()

    return (
        red_generator,
        blue_generator,
    )


# ============================================================================
# EPISODE MANAGER
# ============================================================================

def build_episode_manager(
    training: bool,
) -> EpisodeManager:

    dataset_path = Path(
        config.DATASET_PATH
    )

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------

    metadata_loader = MetadataLoader()

    scenario_loader = ScenarioLoader(
        dataset_path=dataset_path,
        metadata_loader=metadata_loader,
        strict=True,
    )

    path_resolver = PathResolver()

    code_loader = CodeLoader()

    # ------------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------------

    prompt_builder = PromptBuilder()

    # ------------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------------

    (
        red_generator,
        blue_generator,
    ) = build_generators()

    # ------------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------------

    deterministic_checker = (
        DeterministicChecker()
    )

    oracle = Oracle(
        deterministic_checker=(
            deterministic_checker
        )
    )

    interaction_checker = (
        InteractionChecker()
    )

    # ------------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------------

    reward_manager = (
        RewardManager()
    )

    # ------------------------------------------------------------------------
    # Trainers
    #
    # Normal integration mode intentionally has no trainers.
    #
    # This means:
    #
    #     training=False
    #     red_trainer=None
    #     blue_trainer=None
    #
    # and therefore no model weights can be modified.
    # ------------------------------------------------------------------------

    red_trainer = None
    blue_trainer = None

    if training:

        raise NotImplementedError(
            "GRPO update mode is not part of this "
            "integration test. Run without --update."
        )

    # ------------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------------

    replay_buffer = ReplayBuffer(
        capacity=getattr(
            config,
            "REPLAY_BUFFER_CAPACITY",
            None,
        )
    )

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    metrics_logger = MetricsLogger(
        output_dir=Path(
            config.METRICS_PATH
        )
    )

    # ------------------------------------------------------------------------
    # Episode Manager
    # ------------------------------------------------------------------------

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
    )


# ============================================================================
# RESULT DISPLAY
# ============================================================================

def print_episode_result(
    result: Any,
) -> None:

    print()
    print("=" * 70)
    print("EPISODE RESULT")
    print("=" * 70)

    print(
        f"Scenario          : "
        f"{result.scenario_id}"
    )

    print(
        f"Episode ID        : "
        f"{getattr(result, 'episode_id', '')}"
    )

    print()

    print(
        f"Red candidates    : "
        f"{len(result.red_candidates)}"
    )

    print(
        f"Blue candidates   : "
        f"{len(result.blue_candidates)}"
    )

    print(
        f"Top-K defenses    : "
        f"{len(result.top_k_blue_candidates)}"
    )

    print(
        f"Turn-3 challenges : "
        f"{len(result.challenges)}"
    )

    print(
        f"Interactions      : "
        f"{len(result.interaction_results)}"
    )

    print()

    metrics = (
        result.episode_metrics
    )

    for key, value in metrics.items():

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key:<32}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{key:<32}: "
                f"{value}"
            )

    if result.interaction_results:

        print()
        print("INTERACTION RESULTS")
        print("-" * 70)

        for interaction in (
            result.interaction_results
        ):

            print(
                f"Candidate "
                f"{interaction.get('candidate_index', '?')} | "
                f"Case={interaction.get('case', '?')} | "
                f"Red={interaction.get('red_reward', '?')} | "
                f"Blue={interaction.get('blue_reward', '?')}"
            )

    print("=" * 70)


# ============================================================================
# INTEGRATION CHECKS
# ============================================================================

def run_checks(
    result: Any,
    scenario: Any,
) -> bool:

    checks = {
        "scenario loaded":
            result.scenario_id
            == scenario.scenario_id,

        "Red candidates generated":
            len(result.red_candidates)
            > 0,

        "Blue candidates generated":
            len(result.blue_candidates)
            > 0,

        "Top-K defenses selected":
            len(result.top_k_blue_candidates)
            > 0,

        "Turn-3 challenges generated":
            len(result.challenges)
            > 0,

        "Interaction evaluated":
            len(result.interaction_results)
            > 0,

        "Red update disabled":
            result.red_update is None,

        "Blue update disabled":
            result.blue_update is None,
    }

    print()
    print("INTEGRATION CHECKS")
    print("-" * 70)

    passed = True

    for name, status in checks.items():

        print(
            f"[{'PASS' if status else 'FAIL'}] "
            f"{name}"
        )

        if not status:
            passed = False

    return passed


# ============================================================================
# TEST
# ============================================================================

def run_test(
    scenario_id: str,
    perform_update: bool,
) -> bool:

    print("=" * 70)
    print("SOGARL EPISODE INTEGRATION TEST")
    print("=" * 70)

    print(
        f"Scenario : {scenario_id}"
    )

    print(
        f"Updates  : "
        f"{'enabled' if perform_update else 'disabled'}"
    )

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

    config.validate_config()
    config.create_output_directories()

    # ------------------------------------------------------------------------
    # Scenario
    # ------------------------------------------------------------------------

    scenario = load_scenario(
        scenario_id
    )

    print(
        f"Loaded   : "
        f"{scenario.scenario_id}"
    )

    print(
        f"Path     : "
        f"{scenario.scenario_path}"
    )

    # ------------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------------

    episode_manager = (
        build_episode_manager(
            training=perform_update
        )
    )

    # ------------------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------------------

    result = (
        episode_manager.run_episode(
            scenario=scenario,
            training=perform_update,
            episode_id="integration_test",
            epoch=0,
        )
    )

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    print_episode_result(
        result
    )

    # ------------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------------

    passed = run_checks(
        result=result,
        scenario=scenario,
    )

    print()

    print("=" * 70)

    if passed:

        print(
            "EPISODE TEST PASSED"
        )

    else:

        print(
            "EPISODE TEST FAILED"
        )

    print("=" * 70)

    return passed


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    args = parse_args()

    try:

        return (
            0
            if run_test(
                scenario_id=args.scenario,
                perform_update=args.update,
            )
            else 1
        )

    except KeyboardInterrupt:

        print()
        print(
            "Episode test interrupted."
        )

        return 130

    except Exception as exc:

        print()
        print("=" * 70)
        print("EPISODE TEST FAILED")
        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )