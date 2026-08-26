"""
build_primevul_jsonl.py

Main entry point for building a PrimeVul-style JSONL dataset
from the existing scenario-based SFT dataset.

Pipeline
--------

scenario/
    |
    +-- metadata.json
    +-- scenario.md
    +-- red_sft.json
    +-- blue_sft.json       <-- NOT USED
    |
    +-- files/
            |
            v
      MetadataLoader
            |
            v
         Metadata
            |
            +----------------------+
            |                      |
            v                      v
      ScenarioLoader         RedSFTLoader
            |                      |
            v                      v
         Scenario            Red Ground Truth
            |
            v
      PathResolver
            |
            v
      ResolvedFile[]
            |
            v
       CodeLoader
            |
            v
        CodeFile[]
            |
            v
      PromptBuilder
            |
            v
          Prompt
            |
            v
      PrimeVulRecord
            |
            v
      JSONLWriter
            |
            v
  primevul_custom.jsonl


IMPORTANT
---------

The model-visible prompt contains ONLY:

    - system_prompt.txt
    - allowed/neutral scenario.md sections
    - relevant source-code files

The prompt does NOT contain:

    - metadata.json
    - red_sft.json
    - blue_sft.json
    - target
    - CWE
    - CVE
    - expected attack
    - Red reasoning
    - wrong actions
    - injection point
    - sink
    - ground-truth vulnerability information
"""


from pathlib import Path
import sys
import traceback
from typing import Any, Optional

import config

from builders.logger import Logger

from builders.models import (
    Metadata,
    PrimeVulRecord,
)

from builders.metadata_loader import (
    MetadataLoader,
)

from builders.scenario_loader import (
    ScenarioLoader,
)

from builders.red_sft_loader import (
    RedSFTLoader,
)

from builders.path_resolver import (
    PathResolver,
)

from builders.code_loader import (
    CodeLoader,
)

from builders.prompt_builder import (
    PromptBuilder,
)

from builders.jsonl_writer import (
    JSONLWriter,
)

from utils.file_utils import (
    get_scenario_directories,
    read_text,
    ensure_directory,
)

from utils.context_estimator import (
    ContextEstimator,
)


# ============================================================
# PRIMEVUL BUILDER
# ============================================================


class PrimeVulBuilder:
    """
    Orchestrates the complete PrimeVul dataset-building process.
    """

    def __init__(
        self,
        logger=None,
    ):
        self.logger = logger

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        self.metadata_loader = (
            MetadataLoader(
                logger=logger
            )
        )

        # --------------------------------------------------------
        # Scenario
        # --------------------------------------------------------

        self.scenario_loader = (
            ScenarioLoader(
                logger=logger,
                allowed_sections=(
                    config.ALLOWED_SCENARIO_SECTIONS
                ),
                excluded_sections=(
                    config.EXCLUDED_SCENARIO_SECTIONS
                ),
            )
        )

        # --------------------------------------------------------
        # Red SFT
        # --------------------------------------------------------

        self.red_sft_loader = (
            RedSFTLoader(
                logger=logger,
                strict_attack_consistency=True,
            )
        )

        # --------------------------------------------------------
        # Paths
        # --------------------------------------------------------

        self.path_resolver = (
            PathResolver(
                logger=logger
            )
        )

        # --------------------------------------------------------
        # Source code
        # --------------------------------------------------------

        self.code_loader = (
            CodeLoader(
                logger=logger
            )
        )

        # --------------------------------------------------------
        # Prompt
        # --------------------------------------------------------

        self.prompt_builder = (
            PromptBuilder(
                system_prompt_path=(
                    config.SYSTEM_PROMPT_FILE
                ),
                prompt_template_path=(
                    config.PROMPT_TEMPLATE_FILE
                ),
                logger=logger,
            )
        )

        # --------------------------------------------------------
        # Context estimator
        # --------------------------------------------------------

        self.context_estimator = (
            ContextEstimator(
                logger=logger
            )
        )

        # --------------------------------------------------------
        # JSONL writer
        # --------------------------------------------------------

        self.writer = (
            JSONLWriter(
                output_path=(
                    config.OUTPUT_FILE
                ),
                logger=logger,
                overwrite=True,
            )
        )

    # ============================================================
    # BUILD DATASET
    # ============================================================

    def build(self) -> Path:
        """
        Process every scenario and write the final JSONL file.
        """

        self._validate_configuration()

        scenario_directories = (
            get_scenario_directories(
                config.DATASET_ROOT
            )
        )

        if not scenario_directories:
            raise RuntimeError(
                "No scenario directories found in: "
                f"{config.DATASET_ROOT}"
            )

        if config.SORT_SCENARIOS:
            scenario_directories = sorted(
                scenario_directories,
                key=lambda path: (
                    path.name.lower()
                ),
            )

        if self.logger:

            self.logger.info(
                "=================================================="
            )

            self.logger.info(
                "Starting PrimeVul JSONL build"
            )

            self.logger.info(
                f"Dataset root: "
                f"{config.DATASET_ROOT}"
            )

            self.logger.info(
                f"Scenarios found: "
                f"{len(scenario_directories)}"
            )

            self.logger.info(
                "=================================================="
            )

        records = []

        for scenario_path in (
            scenario_directories
        ):

            try:

                record = (
                    self._process_scenario(
                        scenario_path
                    )
                )

                if record is not None:
                    records.append(
                        record
                    )

            except Exception as exc:

                error_message = (
                    f"{scenario_path.name}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                if self.logger:

                    self.logger.error(
                        error_message
                    )

                    self.logger.debug(
                        traceback.format_exc()
                    )

                if not config.CONTINUE_ON_ERROR:
                    raise

        # --------------------------------------------------------
        # No records
        # --------------------------------------------------------

        if not records:

            raise RuntimeError(
                "No valid PrimeVul records were generated."
            )

        # --------------------------------------------------------
        # Write
        # --------------------------------------------------------

        output_path = (
            self.writer.write(
                records
            )
        )

        # --------------------------------------------------------
        # Validate
        # --------------------------------------------------------

        validation = (
            self.writer.validate_existing_file(
                output_path
            )
        )

        if not validation["valid"]:

            raise RuntimeError(
                "Generated JSONL failed validation:\n"
                + "\n".join(
                    validation["errors"]
                )
            )

        # --------------------------------------------------------
        # Final statistics
        # --------------------------------------------------------

        if self.logger:

            self.logger.info(
                "=================================================="
            )

            self.logger.info(
                "PrimeVul JSONL build completed"
            )

            self.logger.info(
                f"Output: {output_path}"
            )

            self.logger.info(
                f"Total records: "
                f"{validation['total_records']}"
            )

            self.logger.info(
                f"Vulnerable: "
                f"{validation['vulnerable']}"
            )

            self.logger.info(
                f"Safe: "
                f"{validation['safe']}"
            )

            self.logger.info(
                "=================================================="
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
        Process one scenario.
        """

        scenario_name = (
            scenario_path.name
        )

        if self.logger:

            self.logger.info(
                f"Processing scenario: "
                f"{scenario_name}"
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
        # 2. LOAD SCENARIO
        # ========================================================

        scenario = (
            self.scenario_loader.load(
                scenario_path
            )
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

        # ========================================================
        # 4. DETERMINE TARGET
        # ========================================================

        target = (
            self._determine_target(
                metadata=metadata,
                red_examples=red_examples,
                scenario_path=scenario_path,
            )
        )

        # ========================================================
        # 5. RESOLVE RELEVANT SOURCE FILES
        # ========================================================

        resolved_files = (
            self.path_resolver.resolve_relevant_files(
                scenario_path,
                metadata,
            )
        )

        # ========================================================
        # 6. LOAD SOURCE CODE
        # ========================================================

        code_files = (
            self.code_loader.load_files(
                resolved_files
            )
        )

        if not code_files:

            raise ValueError(
                f"No relevant source-code files were "
                f"loaded for scenario '{scenario_name}'."
            )

        # ========================================================
        # 7. BUILD MODEL PROMPT
        # ========================================================

        prompt = (
            self.prompt_builder.build(
                scenario=scenario,
                code_files=code_files,
            )
        )

        # ========================================================
        # 8. ESTIMATE CONTEXT SIZE
        # ========================================================

        if config.ENABLE_CONTEXT_ESTIMATION:

            statistics = (
                self.context_estimator.estimate_prompt(
                    system_prompt=prompt.system,
                    user_prompt=prompt.user,
                )
            )

            self.context_estimator.log_statistics(
                statistics=statistics,
                scenario_name=scenario_name,
            )

            if (
                statistics["status"]
                == "OVER_LIMIT"
            ):

                if self.logger:

                    self.logger.warning(
                        f"{scenario_name}: "
                        f"estimated prompt size is "
                        f"above the configured limit."
                    )

            # ----------------------------------------------------
            # Save exact model-visible prompt.
            # ----------------------------------------------------

            if config.SAVE_DEBUG_PROMPTS:

                self._save_debug_prompt(
                    scenario_name=scenario_name,
                    prompt=prompt,
                    statistics=statistics,
                )

        # ========================================================
        # 9. CREATE PRIMEVUL RECORD
        # ========================================================

        record = (
            self._create_record(
                scenario_path=scenario_path,
                metadata=metadata,
                target=target,
                code_files=code_files,
            )
        )

        if self.logger:

            self.logger.info(
                f"Completed: "
                f"{scenario_name} | "
                f"target={target} | "
                f"source_files={len(code_files)}"
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

        Target:

            1 = vulnerable
            0 = not vulnerable

        IMPORTANT
        ---------
        The metadata.json is authoritative.

        We DO NOT use:

            red_sft.json existence
            red reasoning
            attack existence
            absence of a vulnerability field

        as an automatic label.

        We specifically look for an explicit vulnerability
        declaration in metadata.

        Therefore:

            explicit vulnerable
                -> 1

            explicit NOT vulnerable
                -> 0

            ambiguous
                -> ERROR

        This prevents silent label contamination.
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
        # 1. Check explicit boolean vulnerability fields.
        # --------------------------------------------------------

        explicit_boolean = (
            self._find_explicit_boolean_status(
                raw_data
            )
        )

        if explicit_boolean is not None:

            if explicit_boolean:
                return config.VULNERABLE_LABEL

            return config.NOT_VULNERABLE_LABEL

        # --------------------------------------------------------
        # 2. Check explicit textual vulnerability status.
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
        # 3. Fall back to explicit vulnerability objects.
        #
        # This is ONLY used when the metadata schema represents
        # vulnerability information as an explicit non-empty
        # vulnerabilities list.
        # --------------------------------------------------------

        vulnerabilities = getattr(
            metadata,
            "vulnerabilities",
            None,
        )

        if isinstance(
            vulnerabilities,
            list,
        ) and vulnerabilities:

            return config.VULNERABLE_LABEL

        # --------------------------------------------------------
        # 4. Explicit optimal attack.
        #
        # An explicit optimal attack means this scenario has a
        # defined vulnerability/attack ground truth.
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

            return config.VULNERABLE_LABEL

        # --------------------------------------------------------
        # 5. Do NOT use Red SFT to infer target.
        # --------------------------------------------------------

        # red_examples intentionally not used here.
        #
        # Red SFT is validation/supporting information.
        # It is not the label source.
        #
        # This prevents:
        #
        #     red_sft.json exists
        #             ↓
        #          target = 1
        #
        # which would be wrong for your safe scenarios if they
        # also contain Red SFT examples.

        scenario_name = (
            scenario_path.name
        )

        raise ValueError(
            "Could not determine an explicit vulnerability "
            f"label for scenario '{scenario_name}'. "
            "metadata.json must explicitly state whether "
            "the scenario is vulnerable or not vulnerable."
        )

    # ============================================================
    # FIND EXPLICIT BOOLEAN STATUS
    # ============================================================

    def _find_explicit_boolean_status(
        self,
        data: dict[str, Any],
    ) -> Optional[bool]:
        """
        Search metadata recursively for explicit boolean
        vulnerability-status fields.

        Supported examples:

            {
                "vulnerable": true
            }

            {
                "is_vulnerable": false
            }

            {
                "has_vulnerability": false
            }

            {
                "contains_vulnerability": false
            }

        Only recognized field names are considered.
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
    # FIND EXPLICIT TEXT STATUS
    # ============================================================

    def _find_explicit_status(
        self,
        data: dict[str, Any],
    ) -> Optional[str]:
        """
        Search metadata recursively for an explicit textual
        vulnerability status.

        Recognized examples:

            "vulnerability_status": "not vulnerable"

            "status": "safe"

            "vulnerability": "none"

            "vulnerability": "not present"

        Only values attached to vulnerability/status-related
        keys are considered.
        """

        status_keys = {
            "vulnerability_status",
            "vulnerability_state",
            "vulnerability",
            "vulnerability_type",
            "security_status",
            "status",
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
    # NORMALIZE TEXT STATUS
    # ============================================================

    def _normalize_vulnerability_status(
        self,
        value: str,
    ) -> Optional[str]:
        """
        Normalize explicit vulnerability-status text.

        Returns:

            vulnerable
            not_vulnerable
            None

        IMPORTANT:
        This function only recognizes explicit statements.
        It does not interpret an absent field as "safe".
        """

        normalized = (
            value
            .strip()
            .lower()
        )

        normalized = (
            normalized
            .replace("_", " ")
            .replace("-", " ")
        )

        normalized = " ".join(
            normalized.split()
        )

        # --------------------------------------------------------
        # Explicit negative states
        # --------------------------------------------------------

        negative_values = {
            "not vulnerable",
            "non vulnerable",
            "non-vulnerable",
            "no vulnerability",
            "no vulnerabilities",
            "vulnerability absent",
            "vulnerability not present",
            "vulnerability is absent",
            "vulnerability is not present",
            "safe",
            "secure",
            "benign",
            "none",
            "false",
            "no",
        }

        if normalized in negative_values:

            return "not_vulnerable"

        # --------------------------------------------------------
        # Explicit positive states
        # --------------------------------------------------------

        positive_values = {
            "vulnerable",
            "vulnerability present",
            "vulnerability exists",
            "vulnerability detected",
            "vulnerability confirmed",
            "yes",
            "true",
        }

        if normalized in positive_values:

            return "vulnerable"

        return None

    # ============================================================
    # CREATE PRIMEVUL RECORD
    # ============================================================

    def _create_record(
        self,
        scenario_path: Path,
        metadata: Metadata,
        target: int,
        code_files: list,
    ) -> PrimeVulRecord:
        """
        Create the final PrimeVul-style record.
        """

        func = (
            self._build_function_context(
                code_files
            )
        )

        cwe = getattr(
            metadata,
            "cwe_reference",
            None,
        )

        description = getattr(
            metadata,
            "description",
            None,
        )

        project = (
            getattr(
                metadata,
                "scenario_name",
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
            cve_desc=description,
            scenario_id=scenario_path.name,
        )

    # ============================================================
    # BUILD SOURCE-CODE CONTEXT
    # ============================================================

    def _build_function_context(
        self,
        code_files: list,
    ) -> str:
        """
        Combine all relevant source files into the PrimeVul
        `func` field.

        Each file is explicitly delimited.
        """

        blocks = []

        for code_file in code_files:

            relative_path = (
                code_file.relative_path
            )

            content = (
                code_file.content
            )

            blocks.append(
                f"## FILE: {relative_path}\n\n"
                f"{content}"
            )

        if not blocks:

            raise ValueError(
                "Cannot construct PrimeVul `func`: "
                "no source code was loaded."
            )

        return "\n\n".join(
            blocks
        )

    # ============================================================
    # SAVE DEBUG PROMPT
    # ============================================================

    def _save_debug_prompt(
        self,
        scenario_name: str,
        prompt,
        statistics: dict,
    ) -> None:
        """
        Save the exact prompt presented to the model.

        IMPORTANT:
        This debug file contains model-visible information only.
        """

        debug_directory = (
            ensure_directory(
                config.DEBUG_DIR
            )
        )

        safe_name = (
            self._safe_name(
                scenario_name
            )
        )

        prompt_path = (
            debug_directory
            / f"{safe_name}.txt"
        )

        content = (
            "==================================================\n"
            "SYSTEM PROMPT\n"
            "==================================================\n\n"
            f"{prompt.system}\n\n"

            "==================================================\n"
            "USER PROMPT\n"
            "==================================================\n\n"
            f"{prompt.user}\n\n"

            "==================================================\n"
            "ESTIMATED CONTEXT STATISTICS\n"
            "==================================================\n\n"
            f"{statistics}\n"
        )

        prompt_path.write_text(
            content,
            encoding="utf-8",
        )

    # ============================================================
    # SAFE DEBUG NAME
    # ============================================================

    def _safe_name(
        self,
        value: str,
    ) -> str:
        """
        Make a safe filename for debug output.
        """

        unsafe = (
            '\\/:*?"<>|'
        )

        result = value

        for character in unsafe:

            result = result.replace(
                character,
                "_",
            )

        return result.strip()

    # ============================================================
    # CONFIGURATION VALIDATION
    # ============================================================

    def _validate_configuration(
        self,
    ) -> None:
        """
        Validate required paths and prompt structure before
        starting the build.
        """

        # --------------------------------------------------------
        # Dataset
        # --------------------------------------------------------

        if not config.DATASET_ROOT.exists():

            raise FileNotFoundError(
                "DATASET_ROOT does not exist: "
                f"{config.DATASET_ROOT}"
            )

        if not config.DATASET_ROOT.is_dir():

            raise ValueError(
                "DATASET_ROOT is not a directory: "
                f"{config.DATASET_ROOT}"
            )

        # --------------------------------------------------------
        # System prompt
        # --------------------------------------------------------

        if not config.SYSTEM_PROMPT_FILE.exists():

            raise FileNotFoundError(
                "System prompt not found: "
                f"{config.SYSTEM_PROMPT_FILE}"
            )

        # --------------------------------------------------------
        # Prompt template
        # --------------------------------------------------------

        if not config.PROMPT_TEMPLATE_FILE.exists():

            raise FileNotFoundError(
                "Prompt template not found: "
                f"{config.PROMPT_TEMPLATE_FILE}"
            )

        # --------------------------------------------------------
        # Prompt placeholders
        # --------------------------------------------------------

        template = read_text(
            config.PROMPT_TEMPLATE_FILE
        )

        required_placeholders = {
            "{{SCENARIO}}",
            "{{SOURCE_CODE}}",
        }

        missing = [
            placeholder
            for placeholder in (
                required_placeholders
            )
            if placeholder not in template
        ]

        if missing:

            raise ValueError(
                "Prompt template is missing required "
                f"placeholder(s): {missing}"
            )

        # --------------------------------------------------------
        # Output
        # --------------------------------------------------------

        ensure_directory(
            config.OUTPUT_DIR
        )


# ============================================================
# MAIN
# ============================================================


def main() -> int:
    """
    Command-line entry point.
    """

    logger = Logger()

    try:

        builder = (
            PrimeVulBuilder(
                logger=logger
            )
        )

        output_path = (
            builder.build()
        )

        print(
            "\nPrimeVul dataset created:"
        )

        print(
            output_path
        )

        return 0

    except KeyboardInterrupt:

        logger.warning(
            "Build interrupted by user."
        )

        return 130

    except Exception as exc:

        logger.error(
            f"Build failed: {exc}"
        )

        logger.debug(
            traceback.format_exc()
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )