"""
Integration smoke test for SOGARL.

Tests the complete inference pipeline without training.

Checks:

    ScenarioLoader
    MetadataLoader
    PathResolver
    CodeLoader
    ScenarioContext
    PromptBuilder
    Red Generator
    Oracle
    Blue Generator
    Red Turn-3 challenge
    Interaction evaluation

No GRPO update is performed.
No optimizer step is performed.
No checkpoint is saved.
"""

from __future__ import annotations

import sys
from pathlib import Path


# ============================================================================
# PROJECT ROOT
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# IMPORTS
# ============================================================================

from configs import config

from src.data.scenario_loader import ScenarioLoader
from src.data.metadata_loader import MetadataLoader
from src.data.path_resolver import PathResolver
from src.data.code_loader import CodeLoader

from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator

from src.oracle.oracle import Oracle
from src.oracle.interaction_checker import InteractionChecker


# ============================================================================
# HELPERS
# ============================================================================

def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def success(message: str) -> None:
    print(f"[PASS] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise RuntimeError(message)


# ============================================================================
# FIND SCENARIO
# ============================================================================

def find_scenario() -> Path:

    dataset_path = Path(
        config.DATASET_PATH
    )

    if not dataset_path.exists():
        fail(
            f"Dataset path does not exist: "
            f"{dataset_path}"
        )

    scenarios = sorted(
        path
        for path in dataset_path.iterdir()
        if path.is_dir()
        and path.name.startswith("scenario_")
    )

    if not scenarios:
        fail(
            f"No scenario directories found in "
            f"{dataset_path}"
        )

    return scenarios[0]


# ============================================================================
# MAIN TEST
# ============================================================================

def main() -> int:

    try:

        # ------------------------------------------------------------------
        # CONFIG
        # ------------------------------------------------------------------

        section("1. CONFIGURATION")

        config.validate_config()

        success(
            f"Configuration loaded"
        )

        info(
            f"Dataset: {config.DATASET_PATH}"
        )

        info(
            f"Rollouts: {config.NUM_ROLLOUTS}"
        )

        # ------------------------------------------------------------------
        # COMPONENTS
        # ------------------------------------------------------------------

        section("2. INITIALIZE COMPONENTS")

        metadata_loader = MetadataLoader()

        scenario_loader = ScenarioLoader(
            dataset_path=Path(
                config.DATASET_PATH
            ),
            metadata_loader=metadata_loader,
        )

        path_resolver = PathResolver()

        code_loader = CodeLoader()

        prompt_builder = PromptBuilder()

        interaction_checker = (
            InteractionChecker()
        )

        oracle = Oracle()

        success(
            "Data, prompt, and Oracle components initialized"
        )

        # ------------------------------------------------------------------
        # SCENARIO
        # ------------------------------------------------------------------

        section("3. SCENARIO LOADING")

        scenario_path = find_scenario()

        info(
            f"Testing: {scenario_path}"
        )

        scenario = (
            scenario_loader.load_scenario(
                scenario_path
            )
        )

        success(
            f"Scenario loaded: "
            f"{scenario.scenario_id}"
        )

        info(
            f"Description length: "
            f"{len(scenario.scenario_description)}"
        )

        info(
            f"Metadata fields: "
            f"{len(scenario.metadata)}"
        )

        # ------------------------------------------------------------------
        # METADATA
        # ------------------------------------------------------------------

        section("4. METADATA")

        info(
            f"Relevant files: "
            f"{len(scenario.relevant_files)}"
        )

        info(
            f"Noise files available: "
            f"{len(scenario.noise_files)}"
        )

        if not scenario.relevant_files:
            fail(
                "No relevant files found in metadata."
            )

        success(
            "Metadata loaded correctly"
        )

        # ------------------------------------------------------------------
        # PATH RESOLUTION
        # ------------------------------------------------------------------

        section("5. PATH RESOLUTION")

        resolved_files = (
            path_resolver.resolve(
                scenario_path=scenario.scenario_path,
                relevant_files=scenario.relevant_files,
                noise_files=scenario.noise_files,
            )
        )

        success(
            f"Resolved {len(resolved_files)} files"
        )

        relevant_count = sum(
            file.is_relevant
            for file in resolved_files
        )

        noise_count = sum(
            file.is_noise
            for file in resolved_files
        )

        info(
            f"Relevant selected: "
            f"{relevant_count}"
        )

        info(
            f"Noise selected: "
            f"{noise_count}"
        )

        # ------------------------------------------------------------------
        # CODE LOADING
        # ------------------------------------------------------------------

        section("6. CODE LOADING")

        code_files = (
            code_loader.load(
                resolved_files
            )
        )

        if not code_files:
            fail(
                "No code files were loaded."
            )

        success(
            f"Loaded {len(code_files)} files"
        )

        for code_file in code_files:

            print(
                f"  [{code_file.file_type}] "
                f"{code_file.relative_path} "
                f"({code_file.line_count} lines)"
            )

        # ------------------------------------------------------------------
        # CONTEXT
        # ------------------------------------------------------------------

        section("7. SCENARIO CONTEXT")

        from src.data.models import (
            ScenarioContext,
        )

        context = ScenarioContext(
            scenario=scenario,
            code_files=code_files,
        )

        success(
            "ScenarioContext created"
        )

        info(
            f"Total files: "
            f"{context.total_files}"
        )

        info(
            f"Relevant files: "
            f"{len(context.relevant_code_files)}"
        )

        info(
            f"Noise files: "
            f"{len(context.noise_code_files)}"
        )

        # ------------------------------------------------------------------
        # RED PROMPT
        # ------------------------------------------------------------------

        section("8. RED TURN 1 PROMPT")

        red_prompt = (
            prompt_builder
            .build_red_attack_prompt(
                context
            )
        )

        if not red_prompt.strip():
            fail(
                "Red prompt is empty."
            )

        success(
            "Red attack prompt generated"
        )

        print()
        print(
            red_prompt[:2000]
        )

        # ------------------------------------------------------------------
        # RED GENERATOR
        # ------------------------------------------------------------------

        section("9. RED GENERATION")

        info(
            "Loading Red model..."
        )

        red_generator = (
            Generator.from_config(
                role="red"
            )
        )

        success(
            "Red generator initialized"
        )

        red_generations = (
            red_generator.generate_n(
                prompt=red_prompt,
                n=config.NUM_ROLLOUTS,
                temperature=(
                    config.RED_TEMPERATURE
                ),
                metadata={
                    "turn": 1,
                    "generation_type": (
                        "red_attack"
                    ),
                },
            )
        )

        if not red_generations:
            fail(
                "Red produced no generations."
            )

        success(
            f"Red produced "
            f"{len(red_generations)} candidates"
        )

        for generation in red_generations:

            print()
            print(
                f"--- Red Candidate "
                f"{generation.generation_index} ---"
            )

            print(
                generation.response[:1000]
            )

        # ------------------------------------------------------------------
        # RED ORACLE
        # ------------------------------------------------------------------

        section("10. RED ORACLE")

        red_results = []

        for generation in red_generations:

            result = (
                oracle.score_attack(
                    response=generation.response,
                    scenario=scenario,
                )
            )

            red_results.append(
                (
                    generation,
                    result,
                )
            )

            print(
                f"Candidate "
                f"{generation.generation_index}: "
                f"{result.reward:.4f}"
            )

        success(
            "Red Oracle scoring completed"
        )

        # ------------------------------------------------------------------
        # BEST RED ATTACK
        # ------------------------------------------------------------------

        best_red_generation, best_red_result = max(
            red_results,
            key=lambda item: item[1].reward,
        )

        info(
            f"Best Red attack reward: "
            f"{best_red_result.reward:.4f}"
        )

        # ------------------------------------------------------------------
        # BLUE PROMPT
        # ------------------------------------------------------------------

        section("11. BLUE TURN 2 PROMPT")

        blue_prompt = (
            prompt_builder
            .build_blue_defense_prompt(
                context=context,
                attack_best=(
                    best_red_generation.response
                ),
                confidence_passed=True,
            )
        )

        if not blue_prompt.strip():
            fail(
                "Blue prompt is empty."
            )

        success(
            "Blue defense prompt generated"
        )

        print()
        print(
            blue_prompt[:2000]
        )

        # ------------------------------------------------------------------
        # BLUE GENERATOR
        # ------------------------------------------------------------------

        section("12. BLUE GENERATION")

        info(
            "Loading Blue model..."
        )

        blue_generator = (
            Generator.from_config(
                role="blue"
            )
        )

        success(
            "Blue generator initialized"
        )

        blue_generations = (
            blue_generator.generate_n(
                prompt=blue_prompt,
                n=config.NUM_ROLLOUTS,
                temperature=(
                    config.BLUE_TEMPERATURE
                ),
                metadata={
                    "turn": 2,
                    "generation_type": (
                        "blue_defense"
                    ),
                },
            )
        )

        if not blue_generations:
            fail(
                "Blue produced no generations."
            )

        success(
            f"Blue produced "
            f"{len(blue_generations)} candidates"
        )

        for generation in blue_generations:

            print()
            print(
                f"--- Blue Candidate "
                f"{generation.generation_index} ---"
            )

            print(
                generation.response[:1000]
            )

        # ------------------------------------------------------------------
        # BLUE ORACLE
        # ------------------------------------------------------------------

        section("13. BLUE ORACLE")

        blue_results = []

        for generation in blue_generations:

            result = (
                oracle.score_defense(
                    response=generation.response,
                    scenario=scenario,
                    attack_response=(
                        best_red_generation.response
                    ),
                )
            )

            blue_results.append(
                (
                    generation,
                    result,
                )
            )

            print(
                f"Candidate "
                f"{generation.generation_index}: "
                f"{result.reward:.4f}"
            )

        success(
            "Blue Oracle scoring completed"
        )

        # ------------------------------------------------------------------
        # TOP-K
        # ------------------------------------------------------------------

        section("14. TOP-K BLUE DEFENSES")

        top_k = sorted(
            blue_results,
            key=lambda item: item[1].reward,
            reverse=True,
        )[:config.TOP_K]

        if not top_k:
            fail(
                "No Blue candidates available."
            )

        for generation, result in top_k:

            print(
                f"Candidate "
                f"{generation.generation_index} | "
                f"Reward = {result.reward:.4f}"
            )

        success(
            f"Selected {len(top_k)} Blue defenses"
        )

        # ------------------------------------------------------------------
        # TURN 3
        # ------------------------------------------------------------------

        section("15. RED TURN 3 CHALLENGE")

        for blue_generation, blue_result in top_k:

            challenge_prompt = (
                prompt_builder
                .build_red_challenge_prompt(
                    context=context,
                    blue_defense=(
                        blue_generation.response
                    ),
                )
            )

            challenge = (
                red_generator.generate_one(
                    prompt=challenge_prompt,
                    temperature=(
                        config.CHALLENGE_TEMPERATURE
                    ),
                    metadata={
                        "turn": 3,
                        "generation_type": (
                            "red_challenge"
                        ),
                        "defense_candidate_index": (
                            blue_generation
                            .generation_index
                        ),
                    },
                )
            )

            print()
            print(
                f"Defense "
                f"{blue_generation.generation_index}"
            )

            print(
                "Challenge:"
            )

            print(
                challenge.response[:1000]
            )

            # --------------------------------------------------------------
            # INTERACTION ORACLE
            # --------------------------------------------------------------

            interaction = (
                oracle.evaluate_interaction(
                    response=challenge.response,
                    scenario=scenario,
                    blue_defense=(
                        blue_generation.response
                    ),
                )
            )

            print(
                f"Interaction result: "
                f"{interaction}"
            )

        success(
            "Turn-3 challenge pipeline completed"
        )

        # ------------------------------------------------------------------
        # FINISHED
        # ------------------------------------------------------------------

        section("INTEGRATION TEST PASSED")

        print(
            "Scenario loading      : PASS"
        )

        print(
            "Metadata loading      : PASS"
        )

        print(
            "File resolution       : PASS"
        )

        print(
            "Code loading          : PASS"
        )

        print(
            "ScenarioContext       : PASS"
        )

        print(
            "Red prompt             : PASS"
        )

        print(
            "Red generation        : PASS"
        )

        print(
            "Red Oracle             : PASS"
        )

        print(
            "Blue prompt            : PASS"
        )

        print(
            "Blue generation       : PASS"
        )

        print(
            "Blue Oracle            : PASS"
        )

        print(
            "Top-K selection       : PASS"
        )

        print(
            "Red Turn-3 challenge  : PASS"
        )

        print(
            "Interaction evaluation: PASS"
        )

        print()
        print(
            "NO TRAINING WAS PERFORMED."
        )

        return 0

    except Exception as exc:

        print()
        print("=" * 70)
        print("INTEGRATION TEST FAILED")
        print("=" * 70)
        print()
        print(type(exc).__name__)
        print(exc)

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )