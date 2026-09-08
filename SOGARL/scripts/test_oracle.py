"""
test_oracle.py

Integration test for the SOGARL Oracle.

Tests:
    - Red Turn-1 attack scoring
    - Blue Turn-2 defense scoring
    - Red Turn-3 interaction evaluation

Default:
    scenario_001

This test does not:
    - train models
    - perform GRPO
    - update checkpoints
    - modify the dataset
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

from src.generation.generator import Generator

from src.oracle.oracle import Oracle
from src.oracle.semantic_judge import SemanticJudge
from src.oracle.deterministic_checks import DeterministicChecker


# ============================================================================
# TEST RESPONSES
# ============================================================================

TEST_RED_ATTACK = """
Attack:
Exploit the vulnerable authorization path to access a resource
belonging to another user.

Root Cause:
The authorization check is missing or incorrectly applied before
the resource is returned.

Relevant Files:
Use the files identified by the scenario metadata as relevant
to the vulnerability.

Reasoning:
The attacker reaches the protected resource through the affected
request flow without satisfying the intended ownership check.
"""


TEST_BLUE_DEFENSE = """
Defense:
Enforce an authorization check immediately before returning the
requested resource.

Root Cause Addressed:
The resource owner must be verified against the authenticated user.

Relevant Files:
Apply the authorization fix in the files responsible for the
request and resource-access flow.

Reasoning:
The server must never rely on authentication alone as proof that
the authenticated user is authorized to access the specific resource.
"""


TEST_RED_CHALLENGE = """
Challenge:
The proposed defense may still leave the vulnerable resource
accessible if another request path reaches the same resource
without passing through the new authorization check.

Reasoning:
Verify whether every relevant path to the protected resource
enforces the authorization boundary.
"""


# ============================================================================
# LOADING
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


def load_test_scenario(
    loader: ScenarioLoader,
    scenario_id: str,
):

    return loader.load_scenario(
        scenario_id
    )


def load_code_context(
    scenario,
):
    """
    Resolve metadata-defined files and load their contents.

    PathResolver expects:
        scenario_path
        relevant_files
        noise_files
    """

    path_resolver = PathResolver()

    resolved_files = (
        path_resolver.resolve(
            scenario_path=(
                scenario.scenario_path
            ),
            relevant_files=(
                scenario.relevant_files
            ),
            noise_files=(
                scenario.noise_files
            ),
        )
    )

    code_loader = CodeLoader()

    code_files = (
        code_loader.load(
            resolved_files
        )
    )

    scenario.code_files = code_files

    return scenario


# ============================================================================
# SEMANTIC MODEL INITIALIZATION
# ============================================================================

def initialize_semantic_model():
    """
    Initialize the Red Generator used by the Oracle's SemanticJudge.

    This works in both modes:

        USE_SHARED_BACKBONE=True
        USE_SHARED_BACKBONE=False

    SemanticJudge will reuse the already-loaded Red Generator backbone
    and will not load another base model.
    """

    print()
    print("=" * 70)
    print("INITIALIZING SHARED SEMANTIC MODEL")
    print("=" * 70)

    red_generator = Generator.from_config(
        role="red"
    )

    red_generator.eval()

    print("Semantic model initialized.")

    return red_generator


# ============================================================================
# RESULT DISPLAY
# ============================================================================

def print_oracle_score(
    title: str,
    result: Any,
) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    reward = getattr(
        result,
        "reward",
        None,
    )

    print(
        f"Reward : {reward}"
    )

    fields = [
        "attack_label_score",
        "defense_label_score",
        "root_cause_score",
        "relevant_files_score",
        "reasoning_score",
        "format_score",
        "deterministic_score",
        "semantic_score",
    ]

    for field in fields:

        if hasattr(
            result,
            field,
        ):

            print(
                f"{field}: "
                f"{getattr(result, field)}"
            )

    metadata = getattr(
        result,
        "metadata",
        None,
    )

    if metadata:

        print()
        print("Metadata:")
        print("-" * 70)

        for key, value in metadata.items():

            print(
                f"{key}: {value}"
            )


def print_interaction_result(
    result: Any,
) -> None:

    print()
    print("=" * 70)
    print("RED TURN-3 INTERACTION RESULT")
    print("=" * 70)

    print(
        f"Red reward   : "
        f"{getattr(result, 'red_reward', 'N/A')}"
    )

    print(
        f"Blue reward  : "
        f"{getattr(result, 'blue_reward', 'N/A')}"
    )

    print(
        f"Case ID      : "
        f"{getattr(result, 'case_id', 'N/A')}"
    )

    print(
        f"Claim type   : "
        f"{getattr(result, 'red_claim_type', 'N/A')}"
    )

    print(
        f"Ground truth : "
        f"{getattr(result, 'ground_truth_state', 'N/A')}"
    )

    attack_claim = getattr(
        result,
        "attack_claim",
        None,
    )

    if attack_claim:

        print(
            f"Attack claim : "
            f"{attack_claim}"
        )

    verified_attack = getattr(
        result,
        "verified_attack",
        None,
    )

    if verified_attack:

        print(
            f"Verified     : "
            f"{verified_attack}"
        )

    explanation = getattr(
        result,
        "explanation",
        "",
    )

    if explanation:

        print()
        print(
            f"Explanation  : "
            f"{explanation}"
        )


# ============================================================================
# RESULT VALIDATION
# ============================================================================

def validate_score(
    result: Any,
    name: str,
) -> bool:

    success = True

    if not hasattr(
        result,
        "reward",
    ):

        print(
            f"ERROR: {name} result has no reward."
        )

        return False

    reward = getattr(
        result,
        "reward",
        None,
    )

    try:

        reward = float(
            reward
        )

    except (
        TypeError,
        ValueError,
    ):

        print(
            f"ERROR: {name} reward is not numeric: "
            f"{reward!r}"
        )

        return False

    if not (
        0.0
        <= reward
        <= 1.0
    ):

        print(
            f"ERROR: {name} reward is outside "
            f"[0, 1]: {reward}"
        )

        success = False

    return success


def validate_interaction(
    result: Any,
) -> bool:

    success = True

    if not hasattr(
        result,
        "red_reward",
    ):

        print(
            "ERROR: Interaction result has "
            "no red_reward."
        )

        success = False

    if not hasattr(
        result,
        "blue_reward",
    ):

        print(
            "ERROR: Interaction result has "
            "no blue_reward."
        )

        success = False

    if hasattr(
        result,
        "red_reward",
    ):

        if result.red_reward not in (
            -1,
            0,
            1,
        ):

            print(
                "ERROR: Invalid Red interaction "
                f"reward: {result.red_reward}"
            )

            success = False

    if hasattr(
        result,
        "blue_reward",
    ):

        if result.blue_reward not in (
            -1,
            0,
            1,
        ):

            print(
                "ERROR: Invalid Blue interaction "
                f"reward: {result.blue_reward}"
            )

            success = False

    return success


# ============================================================================
# ORACLE TEST
# ============================================================================

def run_test(
    scenario_id: str,
) -> bool:

    print("=" * 70)
    print("SOGARL ORACLE TEST")
    print("=" * 70)

    print(
        f"Scenario: {scenario_id}"
    )

    # ------------------------------------------------------------------------
    # Scenario loader
    # ------------------------------------------------------------------------

    scenario_loader = (
        build_scenario_loader()
    )

    scenario = load_test_scenario(
        loader=scenario_loader,
        scenario_id=scenario_id,
    )

    print(
        f"Path    : "
        f"{scenario.scenario_path}"
    )

    # ------------------------------------------------------------------------
    # Code context
    # ------------------------------------------------------------------------

    scenario = load_code_context(
        scenario
    )

    print(
        f"Files   : "
        f"{len(scenario.code_files)}"
    )

    print(
        f"Relevant: "
        f"{len(scenario.relevant_files)}"
    )

    print(
        f"Noise   : "
        f"{len(scenario.noise_files)}"
    )

    # ------------------------------------------------------------------------
    # Semantic model
    # ------------------------------------------------------------------------
    #
    # Turn 3 requires a complete interaction finding.
    #
    # Deterministic checking alone may not be able to establish
    # real_attack_remains.
    #
    # The SemanticJudge explicitly reuses the already-loaded Red
    # Generator so this test works with both shared and independent
    # policy loading.
    # ------------------------------------------------------------------------

    red_generator = (
        initialize_semantic_model()
    )

    semantic_judge = None

    if config.USE_SEMANTIC_ORACLE:

        semantic_judge = SemanticJudge(
            base_generator=red_generator,
        )

    # ------------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------------

    deterministic_checker = (
        DeterministicChecker()
    )

    oracle = Oracle(
        deterministic_checker=(
            deterministic_checker
        ),
        semantic_judge=(
            semantic_judge
        ),
    )

    # ------------------------------------------------------------------------
    # Red Turn 1
    # ------------------------------------------------------------------------

    print()
    print("Testing Red Turn 1...")

    red_result = (
        oracle.score_attack(
            response=TEST_RED_ATTACK,
            scenario=scenario,
        )
    )

    print_oracle_score(
        title="RED ATTACK RESULT",
        result=red_result,
    )

    # ------------------------------------------------------------------------
    # Blue Turn 2
    # ------------------------------------------------------------------------

    print()
    print("Testing Blue Turn 2...")

    blue_result = (
        oracle.score_defense(
            response=TEST_BLUE_DEFENSE,
            scenario=scenario,
            attack_response=TEST_RED_ATTACK,
        )
    )

    print_oracle_score(
        title="BLUE DEFENSE RESULT",
        result=blue_result,
    )

    # ------------------------------------------------------------------------
    # Red Turn 3
    # ------------------------------------------------------------------------

    print()
    print("Testing Red Turn 3...")

    interaction_result = (
        oracle.evaluate_interaction(
            response=TEST_RED_CHALLENGE,
            scenario=scenario,
            blue_defense=TEST_BLUE_DEFENSE,
        )
    )

    print_interaction_result(
        interaction_result
    )

    # ------------------------------------------------------------------------
    # Validate results
    # ------------------------------------------------------------------------

    success = True

    success = (
        validate_score(
            red_result,
            "Red attack",
        )
        and success
    )

    success = (
        validate_score(
            blue_result,
            "Blue defense",
        )
        and success
    )

    success = (
        validate_interaction(
            interaction_result
        )
        and success
    )

    # ------------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)

    if success:

        print(
            "ORACLE TEST PASSED"
        )

    else:

        print(
            "ORACLE TEST FAILED"
        )

    print("=" * 70)

    return success


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Test the SOGARL Oracle "
            "using a real dataset scenario."
        ),
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="scenario_001",
        help=(
            "Scenario to test "
            "(default: scenario_001)."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    args = parse_args()

    try:

        return (
            0
            if run_test(
                args.scenario
            )
            else 1
        )

    except KeyboardInterrupt:

        print(
            "\nOracle test interrupted."
        )

        return 130

    except Exception as exc:

        print()
        print("=" * 70)
        print("ORACLE TEST FAILED")
        print("=" * 70)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())