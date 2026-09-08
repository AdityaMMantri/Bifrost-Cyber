"""
Integration smoke test for SOGARL.

Tests the complete inference pipeline without training.

Compatible with BOTH:

    USE_SHARED_BACKBONE = True
    USE_SHARED_BACKBONE = False

Architecture:

Shared mode
-----------

    One base model
        ├── Red LoRA
        └── Blue LoRA

    Semantic Oracle reuses the same underlying base model
    with policy adapters temporarily disabled.


Independent mode
----------------

    Base model #1 + Red LoRA
    Base model #2 + Blue LoRA

    Semantic Oracle reuses Red's already-loaded base model
    with the Red adapter temporarily disabled.

No third Oracle base model is loaded.

Checks:

    Configuration
    ScenarioLoader
    MetadataLoader
    PathResolver
    CodeLoader
    ScenarioContext
    PromptBuilder
    Red Generator
    Semantic Oracle
    Blue Generator
    Red Turn-1 generation
    Red Oracle scoring
    Blue Turn-2 generation
    Blue Oracle scoring
    Top-K selection
    Red Turn-3 challenge
    Interaction evaluation

IMPORTANT:

    No GRPO update is performed.
    No optimizer step is performed.
    No checkpoint is saved.

Because this is an INFERENCE smoke test, Red and Blue are explicitly
placed in eval mode after loading.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


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
from src.data.models import ScenarioContext

from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator

from src.oracle.oracle import Oracle
from src.oracle.deterministic_checks import DeterministicChecker
from src.oracle.semantic_judge import SemanticJudge


# ============================================================================
# OUTPUT HELPERS
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


def warning(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise RuntimeError(message)


# ============================================================================
# SMALL INTROSPECTION HELPERS
# ============================================================================

def _safe_attr(obj, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _active_adapter(model) -> Optional[str]:

    # Shared-mode Generator normally exposes _RoleModelView.
    shared_model = _safe_attr(
        model,
        "_shared_model",
        None,
    )

    target = (
        shared_model
        if shared_model is not None
        else model
    )

    active = _safe_attr(
        target,
        "active_adapter",
        None,
    )

    if active is None:
        active = _safe_attr(
            target,
            "active_adapters",
            None,
        )

    return str(active)


def _underlying_model_id(generator: Generator) -> int:
    """
    Return the identity of the actual PEFT model.

    In shared mode:

        Red _RoleModelView
            ↓
        shared PeftModel

        Blue _RoleModelView
            ↓
        same shared PeftModel

    In independent mode:

        Red PeftModel
        Blue PeftModel

    This gives us a reliable smoke-test check for whether
    sharing is actually happening.
    """

    model = generator.get_model()

    shared_model = _safe_attr(
        model,
        "_shared_model",
        None,
    )

    if shared_model is not None:
        return id(shared_model)

    return id(model)


def print_generator_diagnostics(
    name: str,
    generator: Generator,
) -> None:

    model = generator.get_model()
    tokenizer = generator.get_tokenizer()

    underlying = _safe_attr(
        model,
        "_shared_model",
        None,
    )

    if underlying is None:
        underlying = model

    info(
        f"{name} role              : "
        f"{generator.get_role()}"
    )

    info(
        f"{name} model name        : "
        f"{generator.model_name}"
    )

    info(
        f"{name} wrapper type      : "
        f"{type(model).__name__}"
    )

    info(
        f"{name} underlying type   : "
        f"{type(underlying).__name__}"
    )

    info(
        f"{name} underlying ID     : "
        f"{id(underlying)}"
    )

    info(
        f"{name} tokenizer         : "
        f"{_safe_attr(tokenizer, 'name_or_path', 'unknown')}"
    )

    info(
        f"{name} vocab size        : "
        f"{_safe_attr(tokenizer, 'vocab_size', 'unknown')}"
    )

    info(
        f"{name} training          : "
        f"{generator.is_training()}"
    )

    info(
        f"{name} active adapter    : "
        f"{_active_adapter(model)}"
    )


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_runtime_configuration() -> None:
    """
    Additional smoke-test validation.

    config.validate_config() remains the main project validator.

    This function only catches configuration combinations that are
    especially confusing during this integration test.
    """

    config.validate_config()

    use_shared = bool(
        getattr(
            config,
            "USE_SHARED_BACKBONE",
            True,
        )
    )

    use_merged = bool(
        getattr(
            config,
            "USE_MERGED_MODEL",
            False,
        )
    )

    use_semantic = bool(
        getattr(
            config,
            "USE_SEMANTIC_ORACLE",
            True,
        )
    )

    info(
        f"Shared backbone          : "
        f"{use_shared}"
    )

    info(
        f"Merged policy mode       : "
        f"{use_merged}"
    )

    info(
        f"Semantic Oracle          : "
        f"{use_semantic}"
    )

    info(
        f"Base model               : "
        f"{config.BASE_MODEL_NAME}"
    )

    info(
        f"Red adapter              : "
        f"{config.RED_ADAPTER_PATH}"
    )

    info(
        f"Blue adapter             : "
        f"{config.BLUE_ADAPTER_PATH}"
    )

    # ------------------------------------------------------------------
    # Precision sanity check
    # ------------------------------------------------------------------

    try:
        import torch

        if (
            torch.cuda.is_available()
            and getattr(config, "USE_BF16", False)
            and not torch.cuda.is_bf16_supported()
        ):
            fail(
                "USE_BF16=True but the active CUDA GPU does not "
                "support BF16. On a Kaggle T4 use "
                "USE_BF16=False and USE_FP16=True."
            )

    except ImportError:
        pass


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
        if (
            path.is_dir()
            and path.name.startswith(
                "scenario_"
            )
        )
    )

    if not scenarios:
        fail(
            "No scenario directories found in "
            f"{dataset_path}"
        )

    return scenarios[0]


# ============================================================================
# POLICY LOADING
# ============================================================================

def load_policy(
    role: str,
) -> Generator:
    """
    Load a SOGARL policy according to the CURRENT configuration.

    This helper deliberately does NOT care whether the configured
    implementation is:

        shared backbone
        independent backbone
        merged-model compatibility mode

    Generator.from_config() remains the single authority for deciding
    which architecture to instantiate.
    """

    info(
        f"Loading {role.capitalize()} policy..."
    )

    generator = Generator.from_config(
        role=role
    )

    # --------------------------------------------------------------
    # This file performs inference only.
    #
    # Generator.from_config() may load trainable LoRA adapters because
    # the same Generator class is used by GRPO training.
    #
    # For this smoke test, however, generation must happen in eval mode.
    # --------------------------------------------------------------

    generator.eval()

    success(
        f"{role.capitalize()} generator initialized"
    )

    print_generator_diagnostics(
        role.capitalize(),
        generator,
    )

    return generator


# ============================================================================
# ORACLE CONSTRUCTION
# ============================================================================

def build_oracle(
    red_generator: Generator,
) -> Oracle:
    """
    Construct Oracle in a way that works for BOTH model layouts.

    We NEVER rely on:

        Generator._shared_model

    from the integration test.

    Instead we explicitly give SemanticJudge the already-loaded Red
    Generator.

    SemanticJudge itself knows how to unwrap:

        Shared mode:
            _RoleModelView -> shared PeftModel

        Independent mode:
            PeftModel -> itself

    During semantic evaluation SemanticJudge temporarily disables the
    attached policy LoRA, therefore Oracle evaluates with the underlying
    base model without loading another 8B model.
    """

    deterministic_checker = (
        DeterministicChecker()
    )

    semantic_judge = None

    if getattr(
        config,
        "USE_SEMANTIC_ORACLE",
        True,
    ):

        info(
            "Initializing Semantic Oracle from "
            "the already-loaded Red model..."
        )

        semantic_judge = SemanticJudge(
            base_generator=red_generator,
        )

        success(
            "Semantic Oracle initialized"
        )

        info(
            "Oracle model source: Red policy backbone "
            "with LoRA disabled during judging"
        )

    else:

        info(
            "Semantic Oracle disabled by configuration"
        )

    oracle = Oracle(
        deterministic_checker=(
            deterministic_checker
        ),
        semantic_judge=(
            semantic_judge
        ),
    )

    return oracle


# ============================================================================
# SHARING ASSERTION
# ============================================================================

def validate_policy_layout(
    red_generator: Generator,
    blue_generator: Generator,
) -> None:
    """
    Verify that the runtime architecture actually agrees with config.

    This catches accidental model sharing and accidental duplicate
    loading immediately.
    """

    use_shared = bool(
        getattr(
            config,
            "USE_SHARED_BACKBONE",
            True,
        )
    )

    use_merged = bool(
        getattr(
            config,
            "USE_MERGED_MODEL",
            False,
        )
    )

    red_model_id = (
        _underlying_model_id(
            red_generator
        )
    )

    blue_model_id = (
        _underlying_model_id(
            blue_generator
        )
    )

    same_model = (
        red_model_id
        == blue_model_id
    )

    info(
        f"Red underlying model ID  : "
        f"{red_model_id}"
    )

    info(
        f"Blue underlying model ID : "
        f"{blue_model_id}"
    )

    info(
        f"Same underlying model    : "
        f"{same_model}"
    )

    # Merged compatibility models are independent model directories.
    if use_merged:

        if same_model:
            fail(
                "Merged-model mode unexpectedly reused the "
                "same underlying model for Red and Blue."
            )

        success(
            "Merged-model policy isolation verified"
        )

        return

    if use_shared:

        if not same_model:
            fail(
                "USE_SHARED_BACKBONE=True, but Red and Blue "
                "do not reference the same underlying PEFT model."
            )

        success(
            "Shared-backbone architecture verified"
        )

    else:

        if same_model:
            fail(
                "USE_SHARED_BACKBONE=False, but Red and Blue "
                "still reference the same underlying model."
            )

        success(
            "Independent-policy architecture verified"
        )


# ============================================================================
# MAIN TEST
# ============================================================================

def main() -> int:

    try:

        # ------------------------------------------------------------------
        # 1. CONFIGURATION
        # ------------------------------------------------------------------

        section(
            "1. CONFIGURATION"
        )

        validate_runtime_configuration()

        success(
            "Configuration loaded"
        )

        info(
            f"Dataset: "
            f"{config.DATASET_PATH}"
        )

        info(
            f"Rollouts: "
            f"{config.NUM_ROLLOUTS}"
        )

        # ------------------------------------------------------------------
        # 2. INITIALIZE DATA COMPONENTS
        # ------------------------------------------------------------------

        section(
            "2. INITIALIZE COMPONENTS"
        )

        metadata_loader = (
            MetadataLoader()
        )

        scenario_loader = (
            ScenarioLoader(
                dataset_path=Path(
                    config.DATASET_PATH
                ),
                metadata_loader=(
                    metadata_loader
                ),
            )
        )

        path_resolver = (
            PathResolver()
        )

        code_loader = (
            CodeLoader()
        )

        prompt_builder = (
            PromptBuilder()
        )

        success(
            "Data and prompt components initialized"
        )

        # ------------------------------------------------------------------
        # 3. LOAD RED
        # ------------------------------------------------------------------

        section(
            "3. LOAD RED POLICY"
        )

        red_generator = (
            load_policy(
                role="red"
            )
        )

        # ------------------------------------------------------------------
        # 4. ORACLE
        # ------------------------------------------------------------------

        section(
            "4. INITIALIZE ORACLE"
        )

        oracle = (
            build_oracle(
                red_generator=(
                    red_generator
                )
            )
        )

        success(
            "Oracle initialized"
        )

        # ------------------------------------------------------------------
        # 5. SCENARIO
        # ------------------------------------------------------------------

        section(
            "5. SCENARIO LOADING"
        )

        scenario_path = (
            find_scenario()
        )

        info(
            f"Testing: "
            f"{scenario_path}"
        )

        scenario_id = (
            scenario_path.name
        )

        scenario = (
            scenario_loader.load_scenario(
                scenario_id
            )
        )

        success(
            f"Scenario loaded: "
            f"{scenario.scenario_id}"
        )

        info(
            "Description length: "
            f"{len(scenario.scenario_description)}"
        )

        info(
            "Metadata fields: "
            f"{len(scenario.metadata)}"
        )

        # ------------------------------------------------------------------
        # 6. METADATA
        # ------------------------------------------------------------------

        section(
            "6. METADATA"
        )

        info(
            "Relevant files: "
            f"{len(scenario.relevant_files)}"
        )

        info(
            "Noise files available: "
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
        # 7. PATH RESOLUTION
        # ------------------------------------------------------------------

        section(
            "7. PATH RESOLUTION"
        )

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

        success(
            f"Resolved "
            f"{len(resolved_files)} files"
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
        # 8. CODE LOADING
        # ------------------------------------------------------------------

        section(
            "8. CODE LOADING"
        )

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
            f"Loaded "
            f"{len(code_files)} files"
        )

        for code_file in code_files:

            print(
                f"  "
                f"[{code_file.file_type}] "
                f"{code_file.relative_path} "
                f"({code_file.line_count} lines)"
            )

        # ------------------------------------------------------------------
        # 9. SCENARIO CONTEXT
        # ------------------------------------------------------------------

        section(
            "9. SCENARIO CONTEXT"
        )

        context = (
            ScenarioContext(
                scenario=scenario,
                code_files=code_files,
            )
        )

        success(
            "ScenarioContext created"
        )

        info(
            f"Total files: "
            f"{context.total_files}"
        )

        info(
            "Relevant files: "
            f"{len(context.relevant_code_files)}"
        )

        info(
            "Noise files: "
            f"{len(context.noise_code_files)}"
        )

        # ------------------------------------------------------------------
        # 10. RED TURN-1 PROMPT
        # ------------------------------------------------------------------

        section(
            "10. RED TURN 1 PROMPT"
        )

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
        # 11. RED GENERATION
        # ------------------------------------------------------------------

        section(
            "11. RED GENERATION"
        )

        # Integration test = inference only.
        red_generator.eval()

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
                turn=1,
                generation_type=(
                    "red_attack"
                ),
            )
        )

        if not red_generations:
            fail(
                "Red produced no generations."
            )

        success(
            "Red produced "
            f"{len(red_generations)} candidates"
        )

        for generation in red_generations:

            print()

            print(
                "--- Red Candidate "
                f"{generation.generation_index} ---"
            )

            print(
                generation.response[:1000]
            )

        # ------------------------------------------------------------------
        # 12. RED ORACLE
        # ------------------------------------------------------------------

        section(
            "12. RED ORACLE"
        )

        red_results = []

        for generation in red_generations:

            result = (
                oracle.score_attack(
                    response=(
                        generation.response
                    ),
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
                "Candidate "
                f"{generation.generation_index}: "
                f"{result.reward:.4f}"
            )

        success(
            "Red Oracle scoring completed"
        )

        # ------------------------------------------------------------------
        # BEST RED ATTACK
        # ------------------------------------------------------------------

        (
            best_red_generation,
            best_red_result,
        ) = max(
            red_results,
            key=lambda item: (
                item[1].reward
            ),
        )

        info(
            "Best Red attack reward: "
            f"{best_red_result.reward:.4f}"
        )

        # ------------------------------------------------------------------
        # 13. BLUE PROMPT
        # ------------------------------------------------------------------

        section(
            "13. BLUE TURN 2 PROMPT"
        )

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
        # 14. LOAD BLUE
        # ------------------------------------------------------------------

        section(
            "14. LOAD BLUE POLICY"
        )

        blue_generator = (
            load_policy(
                role="blue"
            )
        )

        # ------------------------------------------------------------------
        # Validate actual model architecture.
        # ------------------------------------------------------------------

        validate_policy_layout(
            red_generator=red_generator,
            blue_generator=blue_generator,
        )

        # ------------------------------------------------------------------
        # 15. BLUE GENERATION
        # ------------------------------------------------------------------

        section(
            "15. BLUE GENERATION"
        )

        # Explicit because the policy loader may create trainable LoRA.
        blue_generator.eval()

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
                turn=2,
                generation_type=(
                    "blue_defense"
                ),
            )
        )

        if not blue_generations:
            fail(
                "Blue produced no generations."
            )

        success(
            "Blue produced "
            f"{len(blue_generations)} candidates"
        )

        for generation in blue_generations:

            print()

            print(
                "--- Blue Candidate "
                f"{generation.generation_index} ---"
            )

            print(
                generation.response[:1000]
            )

        # ------------------------------------------------------------------
        # 16. BLUE ORACLE
        # ------------------------------------------------------------------

        section(
            "16. BLUE ORACLE"
        )

        blue_results = []

        for generation in blue_generations:

            result = (
                oracle.score_defense(
                    response=(
                        generation.response
                    ),
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
                "Candidate "
                f"{generation.generation_index}: "
                f"{result.reward:.4f}"
            )

        success(
            "Blue Oracle scoring completed"
        )

        # ------------------------------------------------------------------
        # 17. TOP-K
        # ------------------------------------------------------------------

        section(
            "17. TOP-K BLUE DEFENSES"
        )

        top_k = sorted(
            blue_results,
            key=lambda item: (
                item[1].reward
            ),
            reverse=True,
        )[:config.TOP_K]

        if not top_k:
            fail(
                "No Blue candidates available."
            )

        for generation, result in top_k:

            print(
                "Candidate "
                f"{generation.generation_index} | "
                f"Reward = "
                f"{result.reward:.4f}"
            )

        success(
            f"Selected "
            f"{len(top_k)} Blue defenses"
        )

        # ------------------------------------------------------------------
        # 18. RED TURN-3
        # ------------------------------------------------------------------

        section(
            "18. RED TURN 3 CHALLENGE"
        )

        for (
            blue_generation,
            blue_result,
        ) in top_k:

            challenge_prompt = (
                prompt_builder
                .build_red_challenge_prompt(
                    context=context,
                    blue_defense=(
                        blue_generation.response
                    ),
                )
            )

            # ----------------------------------------------------------
            # In shared mode Blue generation has activated Blue's LoRA.
            #
            # Red.generate_one() calls _activate_adapter(), so it will
            # switch the shared PEFT model back to Red automatically.
            #
            # Calling eval() here additionally guarantees inference mode.
            # ----------------------------------------------------------

            red_generator.eval()

            challenge = (
                red_generator.generate_one(
                    prompt=(
                        challenge_prompt
                    ),
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
                    turn=3,
                    generation_type=(
                        "red_challenge"
                    ),
                )
            )

            print()

            print(
                "Defense "
                f"{blue_generation.generation_index}"
            )

            print(
                "Challenge:"
            )

            print(
                challenge.response[:1000]
            )

            # ----------------------------------------------------------
            # INTERACTION ORACLE
            # ----------------------------------------------------------

            interaction = (
                oracle.evaluate_interaction(
                    response=(
                        challenge.response
                    ),
                    scenario=scenario,
                    blue_defense=(
                        blue_generation.response
                    ),
                )
            )

            print(
                "Interaction result: "
                f"{interaction}"
            )

        success(
            "Turn-3 challenge pipeline completed"
        )

        # ------------------------------------------------------------------
        # FINAL DIAGNOSTICS
        # ------------------------------------------------------------------

        section(
            "19. FINAL MODEL DIAGNOSTICS"
        )

        print_generator_diagnostics(
            "Red",
            red_generator,
        )

        print_generator_diagnostics(
            "Blue",
            blue_generator,
        )

        validate_policy_layout(
            red_generator=red_generator,
            blue_generator=blue_generator,
        )

        # ------------------------------------------------------------------
        # FINISHED
        # ------------------------------------------------------------------

        section(
            "INTEGRATION TEST PASSED"
        )

        print(
            "Configuration          : PASS"
        )

        print(
            "Scenario loading       : PASS"
        )

        print(
            "Metadata loading       : PASS"
        )

        print(
            "File resolution        : PASS"
        )

        print(
            "Code loading           : PASS"
        )

        print(
            "ScenarioContext        : PASS"
        )

        print(
            "Red policy             : PASS"
        )

        print(
            "Oracle construction    : PASS"
        )

        print(
            "Red generation         : PASS"
        )

        print(
            "Red Oracle             : PASS"
        )

        print(
            "Blue policy            : PASS"
        )

        print(
            "Policy layout          : PASS"
        )

        print(
            "Blue generation        : PASS"
        )

        print(
            "Blue Oracle            : PASS"
        )

        print(
            "Top-K selection        : PASS"
        )

        print(
            "Red Turn-3 challenge   : PASS"
        )

        print(
            "Interaction evaluation : PASS"
        )

        print()

        print(
            "NO TRAINING WAS PERFORMED."
        )

        return 0

    except Exception as exc:

        print()

        print(
            "=" * 70
        )

        print(
            "INTEGRATION TEST FAILED"
        )

        print(
            "=" * 70
        )

        print()

        print(
            type(exc).__name__
        )

        print(
            exc
        )

        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )