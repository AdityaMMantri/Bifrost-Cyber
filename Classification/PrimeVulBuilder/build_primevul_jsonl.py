"""
build_primevul_jsonl.py

Main entry point for building a PrimeVul-style JSONL dataset
from the existing scenario-based SFT dataset.

MODEL-VISIBLE DATA
------------------

The classification model receives ONLY:

    - system_prompt.txt
    - neutral scenario.md context
    - relevant source-code files
    - selected noise/distractor source-code files

The model does NOT receive:

    - metadata.json
    - red_sft.json
    - blue_sft.json
    - target
    - CWE
    - CVE
    - expected attack
    - expected defense
    - Red reasoning
    - competing hypotheses
    - injection point
    - sink
    - source metadata
    - ground-truth vulnerability descriptions

GROUND TRUTH
------------
Ground truth is handled entirely on the builder side.
The final PrimeVul record contains:

    project
    commit_id
    target
    func
    cwe
    cve
    cve_desc

`func` contains the actual model-visible prompt produced by
PromptBuilder.
"""

from __future__ import annotations

import random
from pathlib import Path
import sys
import traceback
from typing import Any,Optional

import config
from builders.logger import Logger
from builders.models import (Metadata,PrimeVulRecord,Scenario)
from builders.metadata_loader import MetadataLoader
from builders.scenario_loader import ScenarioLoader
from builders.red_sft_loader import RedSFTLoader
from builders.path_resolver import PathResolver
from builders.code_loader import CodeLoader
from builders.prompt_builder import PromptBuilder
from builders.jsonl_writer import JSONLWriter
from utils.file_utils import (
    get_scenario_directories,
    read_text,
    ensure_directory,
)

from utils.context_estimator import ContextEstimator


# ============================================================
# PRIMEVUL BUILDER
# ============================================================


class PrimeVulBuilder:
    """
    Orchestrates the complete PrimeVul dataset-building process.

    Separation of concerns:

        metadata_loader
            -> ground truth / builder-side metadata

        scenario_loader
            -> neutral scenario.md context

        red_sft_loader
            -> Red SFT validation / supporting ground truth

        path_resolver
            -> relevant + selected noise source file paths

        code_loader
            -> source code

        prompt_builder
            -> COMPLETE MODEL-VISIBLE INPUT

        jsonl_writer
            -> PrimeVul JSONL serialization
    """

    def __init__(self,logger: Optional[Logger] = None) -> None:
        self.logger = logger
        self.metadata_loader = MetadataLoader(logger=logger)
        self.scenario_loader = ScenarioLoader(include_allowed_only=True,reject_content_leakage=True)
        self.red_sft_loader = RedSFTLoader(logger=logger,strict_attack_consistency=True)
        self.path_resolver = PathResolver(logger=logger)
        self.code_loader = CodeLoader(logger=logger)
        self.prompt_builder = PromptBuilder(system_prompt_path=(config.SYSTEM_PROMPT_FILE),prompt_template_path=(
                config.PROMPT_TEMPLATE_FILE),logger=logger)
        self.context_estimator = ContextEstimator(logger=logger)
        self.writer = JSONLWriter(output_path=config.OUTPUT_FILE,logger=logger,overwrite=True)

    # BUILD DATASET

    def build(self) -> Path:
        """
        Process every scenario and write the final JSONL file.
        """
        self._validate_configuration()

        # --------------------------------------------------------
        # Reproducible random noise selection.
        #
        # If RANDOM_SEED exists in config.py, use it.
        # Otherwise Python's normal random state is used.
        # --------------------------------------------------------

        random_seed = getattr(config,"RANDOM_SEED",None)
        if random_seed is not None:
            random.seed(random_seed)
        scenario_directories = (get_scenario_directories(config.DATASET_ROOT))

        if not scenario_directories:
            raise RuntimeError("No scenario directories found in: " f"{config.DATASET_ROOT}")
        if config.SORT_SCENARIOS:
            scenario_directories = sorted(scenario_directories,key=lambda path: (path.name.lower()))
        if self.logger:
            self.logger.start_build(config.DATASET_ROOT)
            self.logger.info(f"Scenarios found: "f"{len(scenario_directories)}")
            
        records: list[PrimeVulRecord] = []
        successful = 0
        failed = 0
        for scenario_path in scenario_directories:

            try:

                record = self._process_scenario(scenario_path)
                if record is not None:
                    records.append(record)
                    successful += 1
            except Exception as exc:
                failed += 1
                error_message = (
                    f"{scenario_path.name}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                if self.logger:
                    self.logger.error(error_message)
                    self.logger.debug(traceback.format_exc())
                if not config.CONTINUE_ON_ERROR:
                    raise

        if not records:
            raise RuntimeError("No valid PrimeVul records were generated.")

        output_path = self.writer.write(records)
        # Validate generated JSONL
        validation = (self.writer.validate_existing_file(output_path))
        if not validation["valid"]:
            raise RuntimeError("Generated JSONL failed validation:\n"
                + "\n".join(validation["errors"]))

        # Final statistics

        if self.logger:

            self.logger.build_complete(
                total_scenarios=len(
                    scenario_directories
                ),
                successful=successful,
                failed=failed,
                output_file=output_path,
            )

            self.logger.info(
                f"JSONL records validated: "
                f"{validation['total_records']}"
            )

            self.logger.info(
                f"Vulnerable records: "
                f"{validation['vulnerable']}"
            )

            self.logger.info(
                f"Safe records: "
                f"{validation['safe']}"
            )

        return output_path

    # ============================================================
    # PROCESS ONE SCENARIO
    # ============================================================

    def _process_scenario(
        self,
        scenario_path: Path,
    ) -> PrimeVulRecord:
        """
        Process exactly one scenario.

        The order is intentional:

            metadata
                ↓
            scenario.md
                ↓
            red SFT
                ↓
            target
                ↓
            relevant files
                ↓
            randomly selected noise files
                ↓
            CodeLoader
                ↓
            PromptBuilder
                ↓
            PrimeVulRecord
        """

        scenario_name = (
            scenario_path.name
        )

        if self.logger:

            self.logger.start_scenario(
                scenario_name
            )

        # ========================================================
        # 1. LOAD METADATA
        # ========================================================

        metadata_path = (
            scenario_path
            / "metadata.json"
        )

        metadata = (
            self.metadata_loader.load(
                metadata_path
            )
        )

        # ========================================================
        # 2. LOAD SCENARIO.MD
        # ========================================================

        if not config.INCLUDE_SCENARIO_CONTEXT:

            raise RuntimeError(
                "INCLUDE_SCENARIO_CONTEXT=False. "
                "This dataset requires scenario.md "
                "application context."
            )

        scenario_context = (
            self.scenario_loader.load(
                scenario_path,
                scenario_id=scenario_name,
            )
        )

        if not scenario_context.loaded:

            raise RuntimeError(
                f"scenario.md was not successfully "
                f"loaded for {scenario_name}"
            )

        if not scenario_context.text.strip():

            raise RuntimeError(
                f"scenario.md produced empty "
                f"neutral context for {scenario_name}"
            )

        if self.logger:

            self.logger.info(
                f"Loaded scenario.md: "
                f"{scenario_context.source_file}"
            )

            self.logger.info(
                f"Scenario context: "
                f"{len(scenario_context.text):,} characters"
            )

            self.logger.info(
                f"Included scenario sections: "
                f"{scenario_context.included_section_count}"
            )

            self.logger.info(
                f"Excluded scenario sections: "
                f"{scenario_context.excluded_section_count}"
            )

        # --------------------------------------------------------
        # Convert canonical ScenarioContext into Scenario.
        # --------------------------------------------------------

        scenario = Scenario.from_context(
            scenario_context
        )

        # ========================================================
        # 3. LOAD RED SFT
        # ========================================================

        red_examples = []

        red_sft_path = (
            scenario_path
            / "red_sft.json"
        )

        if config.USE_RED_SFT:

            if not red_sft_path.exists():

                raise FileNotFoundError(
                    f"red_sft.json not found for "
                    f"scenario '{scenario_name}': "
                    f"{red_sft_path}"
                )

            red_examples = (
                self.red_sft_loader.load(
                    red_sft_path
                )
            )

            if self.logger:

                self.logger.info(
                    f"Loaded "
                    f"{len(red_examples)} Red SFT "
                    f"example(s)"
                )

        # ========================================================
        # 4. DETERMINE TARGET
        # ========================================================

        target = self._determine_target(
            metadata=metadata,
            red_examples=red_examples,
            scenario_path=scenario_path,
        )

        # ========================================================
        # 5. SELECT NOISE FILES
        # ========================================================
        #
        # ALL relevant files are retained.
        #
        # From metadata.noise_files we randomly select between
        # MIN_NOISE_FILES and MAX_NOISE_FILES.
        #
        # The model is NOT told which files are noise.
        # ========================================================

        noise_files = list(
            getattr(
                metadata,
                "noise_files",
                [],
            )
            or []
        )

        min_noise_files = int(
            getattr(
                config,
                "MIN_NOISE_FILES",
                2,
            )
        )

        max_noise_files = int(
            getattr(
                config,
                "MAX_NOISE_FILES",
                5,
            )
        )

        if min_noise_files < 0:

            raise ValueError(
                "MIN_NOISE_FILES cannot be negative."
            )

        if max_noise_files < min_noise_files:

            raise ValueError(
                "MAX_NOISE_FILES must be greater than "
                "or equal to MIN_NOISE_FILES."
            )

        if noise_files:

            noise_count = random.randint(
                min_noise_files,
                max_noise_files,
            )

            selected_noise_files = (
                random.sample(
                    noise_files,
                    min(
                        noise_count,
                        len(noise_files),
                    ),
                )
            )

        else:

            selected_noise_files = []

        if self.logger:

            self.logger.info(
                f"Relevant files requested: "
                f"{len(metadata.relevant_files)}"
            )

            self.logger.info(
                f"Noise files available: "
                f"{len(noise_files)}"
            )

            self.logger.info(
                f"Noise files selected: "
                f"{len(selected_noise_files)}"
            )

            if selected_noise_files:

                self.logger.info(
                    "Selected noise files: "
                    + ", ".join(
                        selected_noise_files
                    )
                )

        # ========================================================
        # 6. RESOLVE RELEVANT + NOISE SOURCE FILES
        # ========================================================

        resolved_relevant_files = (
            self.path_resolver.resolve_files(
                scenario_path,
                metadata.relevant_files,
                file_type="relevant",
            )
        )

        resolved_noise_files = (
            self.path_resolver.resolve_files(
                scenario_path,
                selected_noise_files,
                file_type="noise",
            )
        )

        resolved_files = (
            resolved_relevant_files
            + resolved_noise_files
        )

        if not resolved_files:

            raise ValueError(
                f"No source files resolved "
                f"for scenario '{scenario_name}'."
            )

        # ========================================================
        # 7. LOAD SOURCE CODE
        # ========================================================

        code_files = (
            self.code_loader.load(
                resolved_files
            )
        )

        if not code_files:

            raise ValueError(
                f"No source-code files were "
                f"loaded for scenario '{scenario_name}'."
            )

        if self.logger:

            self.logger.files_loaded(
                len(code_files)
            )

            relevant_count = sum(
                1
                for code_file in code_files
                if getattr(
                    code_file,
                    "file_type",
                    "",
                ) == "relevant"
            )

            noise_count_loaded = sum(
                1
                for code_file in code_files
                if getattr(
                    code_file,
                    "file_type",
                    "",
                ) == "noise"
            )

            self.logger.info(
                f"Relevant files in prompt: "
                f"{relevant_count}"
            )

            self.logger.info(
                f"Noise files in prompt: "
                f"{noise_count_loaded}"
            )

        # ========================================================
        # 8. BUILD COMPLETE MODEL-VISIBLE PROMPT
        # ========================================================

        prompt = (
            self.prompt_builder.build(
                scenario=scenario,
                code_files=code_files,
            )
        )

        if prompt is None:

            raise RuntimeError(
                f"PromptBuilder returned None for "
                f"{scenario_name}"
            )

        model_prompt = getattr(
            prompt,
            "prompt",
            None,
        )

        if not isinstance(
            model_prompt,
            str,
        ) or not model_prompt.strip():

            raise RuntimeError(
                f"PromptBuilder generated an empty "
                f"model prompt for {scenario_name}"
            )

        if self.logger:

            self.logger.prompt_created(
                len(model_prompt)
            )

        # ========================================================
        # 9. ESTIMATE CONTEXT SIZE
        # ========================================================

        statistics: dict[str, Any] = {}

        if config.ENABLE_CONTEXT_ESTIMATION:

            statistics = (
                self.context_estimator.estimate_prompt(
                    system_prompt=(
                        self.prompt_builder.system_prompt
                    ),
                    user_prompt=model_prompt,
                )
            )

            self.context_estimator.log_statistics(
                statistics=statistics,
                scenario_name=scenario_name,
            )

            if (
                statistics.get("status")
                == "OVER_LIMIT"
            ):

                if self.logger:

                    self.logger.warning(
                        f"{scenario_name}: "
                        f"estimated prompt size is "
                        f"above the configured limit."
                    )

        # ========================================================
        # 10. SAVE EXACT MODEL-VISIBLE DEBUG PROMPT
        # ========================================================

        if config.SAVE_DEBUG_PROMPTS:

            self._save_debug_prompt(
                scenario_name=scenario_name,
                prompt=prompt,
                statistics=statistics,
            )

        # ========================================================
        # 11. CREATE PRIMEVUL RECORD
        # ========================================================

        record = self._create_record(
            scenario_path=scenario_path,
            metadata=metadata,
            target=target,
            prompt=prompt,
        )

        if self.logger:

            self.logger.scenario_summary(
                scenario_name=scenario_name,
                relevant_files=sum(
                    1
                    for code_file in code_files
                    if getattr(
                        code_file,
                        "file_type",
                        "",
                    ) == "relevant"
                ),
                target=target,
                cwe=self._format_cwe(
                    metadata.cwe_reference
                ),
                code_characters=sum(
                    len(
                        getattr(
                            code_file,
                            "content",
                            "",
                        )
                    )
                    for code_file in code_files
                ),
            )

            self.logger.scenario_complete(
                scenario_name
            )

        return record

    # ============================================================
    # TARGET DETERMINATION
    # ============================================================

    def _determine_target(
        self,
        metadata: Metadata,
        red_examples: list,
        scenario_path: Path,
    ) -> int:
        """
        Determine the binary PrimeVul classification label.

        1 = vulnerable
        0 = non-vulnerable

        Ground truth is builder-side only.

        Red SFT is deliberately NOT used as an automatic target
        source. Its existence does not imply vulnerability.
        """

        raw_data = getattr(
            metadata,
            "raw_data",
            {},
        )

        if not isinstance(
            raw_data,
            dict,
        ):
            raw_data = {}

        # --------------------------------------------------------
        # Explicit boolean vulnerability state
        # --------------------------------------------------------

        explicit_boolean = (
            self._find_explicit_boolean_status(
                raw_data
            )
        )

        if explicit_boolean is not None:

            return (
                config.VULNERABLE_LABEL
                if explicit_boolean
                else config.NOT_VULNERABLE_LABEL
            )

        # --------------------------------------------------------
        # Explicit textual vulnerability state
        # --------------------------------------------------------

        explicit_status = (
            self._find_explicit_status(
                raw_data
            )
        )

        if explicit_status == "vulnerable":

            return config.VULNERABLE_LABEL

        if explicit_status == "not_vulnerable":

            return config.NOT_VULNERABLE_LABEL

        # --------------------------------------------------------
        # Explicit vulnerability objects
        # --------------------------------------------------------

        vulnerabilities = getattr(
            metadata,
            "vulnerabilities",
            None,
        )

        if (
            isinstance(
                vulnerabilities,
                list,
            )
            and vulnerabilities
        ):

            return config.VULNERABLE_LABEL

        # --------------------------------------------------------
        # Explicit optimal attack
        # --------------------------------------------------------

        optimal_attack = getattr(
            metadata,
            "optimal_attack",
            None,
        )

        if (
            isinstance(
                optimal_attack,
                str,
            )
            and optimal_attack.strip()
        ):

            normalized = (
                self._normalize_attack_label(
                    optimal_attack
                )
            )

            if normalized in {
                "none",
                "no_attack",
                "no_exploitable_vulnerability",
                "no_exploitable_vulnerabilities",
                "not_vulnerable",
                "non_vulnerable",
                "safe",
            }:

                return config.NOT_VULNERABLE_LABEL

            return config.VULNERABLE_LABEL

        # --------------------------------------------------------
        # Explicit classification dictionary
        # --------------------------------------------------------

        classification = raw_data.get(
            "classification"
        )

        if isinstance(
            classification,
            dict,
        ):

            classification_value = (
                classification.get(
                    "label"
                )
                or classification.get(
                    "ground_truth"
                )
                or classification.get(
                    "status"
                )
            )

            if isinstance(
                classification_value,
                str,
            ):

                normalized = (
                    self._normalize_attack_label(
                        classification_value
                    )
                )

                if normalized in {
                    "safe",
                    "non_vulnerable",
                    "not_vulnerable",
                    "no_vulnerability",
                    "none",
                }:

                    return (
                        config.NOT_VULNERABLE_LABEL
                    )

                if normalized in {
                    "vulnerable",
                }:

                    return (
                        config.VULNERABLE_LABEL
                    )

        # --------------------------------------------------------
        # DO NOT infer from Red SFT.
        # --------------------------------------------------------

        _ = red_examples

        raise ValueError(
            "Could not determine an explicit vulnerability "
            f"label for scenario '{scenario_path.name}'. "
            "metadata.json must explicitly state whether "
            "the scenario is vulnerable or not vulnerable."
        )

    # ============================================================
    # BOOLEAN STATUS SEARCH
    # ============================================================

    def _find_explicit_boolean_status(
        self,
        data: dict[str, Any],
    ) -> Optional[bool]:
        """
        Recursively find explicit boolean vulnerability state.
        """

        boolean_keys = {
            "vulnerable",
            "is_vulnerable",
            "has_vulnerability",
            "contains_vulnerability",
            "vulnerability_present",
        }

        for key, value in data.items():

            normalized_key = (
                str(key)
                .strip()
                .lower()
            )

            if (
                normalized_key
                in boolean_keys
            ):

                if isinstance(
                    value,
                    bool,
                ):

                    return value

            if isinstance(
                value,
                dict,
            ):

                result = (
                    self._find_explicit_boolean_status(
                        value
                    )
                )

                if result is not None:
                    return result

            elif isinstance(
                value,
                list,
            ):

                for item in value:

                    if isinstance(
                        item,
                        dict,
                    ):

                        result = (
                            self._find_explicit_boolean_status(
                                item
                            )
                        )

                        if result is not None:
                            return result

        return None

    # ============================================================
    # TEXT STATUS SEARCH
    # ============================================================

    def _find_explicit_status(
        self,
        data: dict[str, Any],
    ) -> Optional[str]:
        """
        Recursively find explicit textual vulnerability status.
        """

        status_keys = {
            "vulnerability_status",
            "vulnerability_state",
            "vulnerability",
            "vulnerability_type",
            "security_status",
            "status",
            "classification",
        }

        for key, value in data.items():

            normalized_key = (
                str(key)
                .strip()
                .lower()
            )

            if (
                normalized_key
                in status_keys
                and isinstance(
                    value,
                    str,
                )
            ):

                result = (
                    self._normalize_vulnerability_status(
                        value
                    )
                )

                if result is not None:
                    return result

            if isinstance(
                value,
                dict,
            ):

                result = (
                    self._find_explicit_status(
                        value
                    )
                )

                if result is not None:
                    return result

            elif isinstance(
                value,
                list,
            ):

                for item in value:

                    if isinstance(
                        item,
                        dict,
                    ):

                        result = (
                            self._find_explicit_status(
                                item
                            )
                        )

                        if result is not None:
                            return result

        return None

    # ============================================================
    # NORMALIZE VULNERABILITY STATUS
    # ============================================================

    def _normalize_vulnerability_status(
        self,
        value: str,
    ) -> Optional[str]:
        """
        Normalize explicit vulnerability state.

        Returns:

            vulnerable
            not_vulnerable
            None
        """

        normalized = (
            value
            .strip()
            .lower()
        )

        normalized = (
            normalized
            .replace(
                "_",
                " ",
            )
            .replace(
                "-",
                " ",
            )
        )

        normalized = " ".join(
            normalized.split()
        )

        negative_values = {
            "not vulnerable",
            "non vulnerable",
            "no vulnerability",
            "no vulnerabilities",
            "vulnerability absent",
            "vulnerability not present",
            "safe",
            "secure",
            "benign",
            "none",
            "false",
            "no",
            "no exploitable vulnerability",
            "no exploitable vulnerabilities",
        }

        if normalized in negative_values:

            return "not_vulnerable"

        positive_values = {
            "vulnerable",
            "vulnerability present",
            "vulnerability exists",
            "vulnerability detected",
            "vulnerability confirmed",
            "true",
            "yes",
        }

        if normalized in positive_values:

            return "vulnerable"

        return None

    # ============================================================
    # NORMALIZE ATTACK LABEL
    # ============================================================

    def _normalize_attack_label(
        self,
        value: str,
    ) -> str:
        """
        Normalize an attack/classification label.
        """

        normalized = (
            value
            .strip()
            .lower()
        )

        normalized = (
            normalized
            .replace(
                "_",
                " ",
            )
            .replace(
                "-",
                " ",
            )
        )

        normalized = " ".join(
            normalized.split()
        )

        return normalized.replace(
            " ",
            "_",
        )

    # ============================================================
    # CREATE PRIMEVUL RECORD
    # ============================================================

    def _create_record(self,scenario_path: Path,metadata: Metadata,target: int,prompt) -> PrimeVulRecord:
        """
        Create the final PrimeVul-style record.

        CRITICAL:

        `func` is the exact model-visible prompt produced by
        PromptBuilder.

        This is NOT reconstructed from source code.

        Therefore scenario.md context survives into the final
        JSONL record.
        """

        if prompt is None:

            raise ValueError(f"Prompt is missing for "f"{scenario_path.name}")
        func = getattr(prompt,"prompt",None)
        if not isinstance(func,str) or not func.strip():
            raise ValueError(
                f"PromptBuilder produced an empty "
                f"model input for {scenario_path.name}")

        # --------------------------------------------------------
        # CWE is metadata, not prompt content.
        # --------------------------------------------------------

        cwe = getattr(
            metadata,
            "cwe_reference",
            None,
        )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # custom vulnerability descriptions can contain
        # the exact answer and therefore must not become part of
        # any model-visible field.
        #
        # cve_desc is retained as None unless you later introduce
        # a genuinely CVE-derived description.
        # --------------------------------------------------------

        cve_desc = None

        project = (
            getattr(
                metadata,
                "scenario_name",
                None,
            )
            or getattr(
                metadata,
                "name",
                None,
            )
            or scenario_path.name
        )

        commit_id = (
            getattr(
                metadata,
                "commit_id",
                None,
            )
            or config.DEFAULT_COMMIT_ID
        )

        cve = (
            getattr(
                metadata,
                "cve",
                None,
            )
            or config.DEFAULT_CVE
        )

        return PrimeVulRecord(
            project=project,
            commit_id=commit_id,
            target=target,
            func=func,
            cwe=cwe,
            cve=cve,
            cve_desc=cve_desc,
            scenario_id=scenario_path.name,
        )

    # ============================================================
    # SAVE DEBUG PROMPT
    # ============================================================
    def _save_debug_prompt(self,scenario_name: str,prompt,statistics: dict[str, Any]) -> None:
        """
        Save the exact model-visible prompt.

        This file intentionally contains only:

            SYSTEM PROMPT
            +
            USER PROMPT
            +
            prompt statistics

        It does NOT contain metadata.json or Red SFT data.
        """
        debug_directory = ensure_directory(config.DEBUG_DIR)
        safe_name = self._safe_name(scenario_name)
        prompt_path = (debug_directory/ f"{safe_name}.txt")
        model_prompt = getattr(prompt,"prompt",None)
        if not isinstance(model_prompt,str):
            raise ValueError(f"Cannot save debug prompt for "f"{scenario_name}: invalid prompt.")

        content = (
            "==================================================\n"
            "SYSTEM PROMPT\n"
            "==================================================\n\n"
            f"{self.prompt_builder.system_prompt}\n\n"

            "==================================================\n"
            "USER PROMPT / MODEL INPUT\n"
            "==================================================\n\n"
            f"{model_prompt}\n\n"

            "==================================================\n"
            "ESTIMATED CONTEXT STATISTICS\n"
            "==================================================\n\n"
            f"{statistics}\n"
        )

        prompt_path.write_text(content,encoding="utf-8")
        if self.logger:
            self.logger.info(f"Saved debug prompt: "f"{prompt_path}")

    # ============================================================
    # SAFE DEBUG FILENAME
    # ============================================================

    @staticmethod
    def _safe_name(value: str) -> str:
        """
        Make a safe filesystem name.
        """
        unsafe = ('\\/:*?"<>|')
        result = value
        for character in unsafe:
            result = result.replace(character,"_")
        return result.strip()

    # ============================================================
    # CWE FORMATTER
    # ============================================================
    @staticmethod
    def _format_cwe(cwe: Any) -> Optional[str]:
        """
        Format CWE metadata for logging only.
        """

        if cwe is None:
            return None

        if isinstance(cwe,list):
            values = [str(item) for item in cwe if item is not None]
            return ", ".join(values) or None
        return str(cwe)

    # ============================================================
    # CONFIGURATION VALIDATION
    # ============================================================

    def _validate_configuration(self) -> None:
        """
        Validate required paths before starting the build.
        """

        if not config.DATASET_ROOT.exists():
            raise FileNotFoundError("DATASET_ROOT does not exist: "f"{config.DATASET_ROOT}")
        if not config.DATASET_ROOT.is_dir():
            raise ValueError("DATASET_ROOT is not a directory: "f"{config.DATASET_ROOT}")
        
        # System prompt
        if not config.SYSTEM_PROMPT_FILE.exists():
            raise FileNotFoundError("System prompt not found: "f"{config.SYSTEM_PROMPT_FILE}")
        # Prompt template
        if not config.PROMPT_TEMPLATE_FILE.exists():
            raise FileNotFoundError("Prompt template not found: "f"{config.PROMPT_TEMPLATE_FILE}")
        # Prompt template validation
        template = read_text(config.PROMPT_TEMPLATE_FILE)
        required_placeholders = {
            "{{SCENARIO}}",
            "{{SOURCE_CODE}}",
        }

        missing = [placeholder for placeholder in (required_placeholders) if placeholder not in template]
        if missing:
            raise ValueError("Prompt template is missing required "f"placeholder(s): {missing}")
        # Output directory
        ensure_directory(config.OUTPUT_DIR)
        # Debug directory
        if config.SAVE_DEBUG_PROMPTS:
            ensure_directory(config.DEBUG_DIR)
        # Log directory
        log_file = getattr(config,"LOG_FILE",None)
        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True,exist_ok=True)


def main() -> int:
    logger = Logger(name="PrimeVulBuilder",log_file=config.LOG_FILE)
    try:
        builder = PrimeVulBuilder(logger=logger)
        output_path = builder.build()
        print("\nPrimeVul dataset created:")
        print(output_path)
        return 0

    except KeyboardInterrupt:
        logger.warning("Build interrupted by user.")
        return 130
    
    except Exception as exc:
        logger.error(f"Build failed: {exc}")
        logger.debug(traceback.format_exc())

        return 1


if __name__ == "__main__":
    sys.exit(main())