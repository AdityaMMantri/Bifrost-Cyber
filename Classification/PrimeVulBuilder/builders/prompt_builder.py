from __future__ import annotations

from pathlib import Path

from .models import (
    CodeFile,
    ModelInput,
    Prompt,
    PromptResult,
    Scenario,
    ScenarioContext,
)


class PromptBuilder:
    """
    Builds the canonical model input and final classification prompt.

    DATA FLOW
    ---------

        scenario.md
             |
             v
        ScenarioLoader
             |
             v
        ScenarioContext
             |
             +----------------------+
             |                      |
             v                      v
        PromptBuilder          ModelInput
             |                      |
             v                      |
        PromptResult               |
             |                      |
             +----------+-----------+
                        |
                        v
                  JSONL / Debug


    MODEL-VISIBLE
    -------------
        1. system_prompt.txt
        2. neutral scenario.md context
        3. relevant source-code files


    BUILDER-ONLY
    ------------
        metadata.json
        red_sft.json
        blue_sft.json
        target
        attack labels
        Red-Team reasoning
        CWE
        CVE
        injection point
        sink
        expected attack
        optimal attack
        optimal defense
        vulnerability details


    IMPORTANT
    ---------
    This class never reads metadata.json for prompt content.

    Ground truth must never be inserted into the prompt.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        system_prompt_path: Path,
        prompt_template_path: Path,
        logger=None,
    ):
        self.system_prompt_path = Path(
            system_prompt_path
        )

        self.prompt_template_path = Path(
            prompt_template_path
        )

        self.logger = logger

        # --------------------------------------------------------
        # Load system prompt.
        # --------------------------------------------------------

        self.system_prompt = self._load_template_file(
            self.system_prompt_path,
            "system prompt",
        )

        # --------------------------------------------------------
        # Load prompt template.
        # --------------------------------------------------------

        self.prompt_template = self._load_template_file(
            self.prompt_template_path,
            "prompt template",
        )

        # --------------------------------------------------------
        # Validate immediately.
        # --------------------------------------------------------

        self._validate_template()

    # ============================================================
    # PUBLIC API
    # ============================================================

    def build(
        self,
        scenario: Scenario,
        code_files: list[CodeFile],
    ) -> PromptResult:
        """
        Build the complete model-visible input.

        Returns PromptResult containing:

            prompt
            model_input
            statistics

        The SAME ModelInput object should later be supplied to
        JSONLWriter so that debug output and JSONL cannot drift
        apart.
        """

        if not isinstance(
            scenario,
            Scenario,
        ):
            raise TypeError(
                "scenario must be a Scenario object."
            )

        if not code_files:
            raise ValueError(
                "Cannot build a classification prompt "
                "without source code."
            )

        # --------------------------------------------------------
        # 1. Build neutral scenario context.
        # --------------------------------------------------------

        scenario_context = (
            self._build_scenario_context(
                scenario
            )
        )

        # --------------------------------------------------------
        # 2. Build source code.
        # --------------------------------------------------------

        source_code = (
            self._build_source_code(
                code_files
            )
        )

        # --------------------------------------------------------
        # 3. Determine canonical scenario ID.
        #
        # ScenarioLoader should already have loaded the scenario
        # from the scenario directory. Prefer scenario_id when
        # available, otherwise fall back to scenario_name.
        # --------------------------------------------------------

        scenario_id = self._get_scenario_id(
            scenario
        )

        # --------------------------------------------------------
        # 4. Construct canonical ModelInput.
        #
        # THIS is the critical architectural change.
        #
        # Both:
        #
        #     PromptBuilder
        #     JSONLWriter
        #
        # should use this same object.
        # --------------------------------------------------------

        model_input = ModelInput(
            scenario_id=scenario_id,
            scenario_context=scenario_context,
            source_code=source_code,
            source_files=[
                code_file.relative_path
                for code_file in code_files
                if code_file.content.strip()
            ],
        )

        # --------------------------------------------------------
        # 5. Render user prompt from canonical ModelInput.
        # --------------------------------------------------------

        user_prompt = self._render_template(
            scenario_context=model_input.scenario_context,
            source_code=model_input.source_code,
        )

        # --------------------------------------------------------
        # 6. Construct Prompt.
        # --------------------------------------------------------

        prompt = Prompt(
            system=self.system_prompt,
            user=user_prompt,
        )

        # --------------------------------------------------------
        # 7. Estimate tokens.
        #
        # This remains a rough estimate. Exact tokenization should
        # be performed by the model tokenizer later.
        # --------------------------------------------------------

        estimated_tokens = (
            self._estimate_tokens(
                prompt
            )
        )

        status = (
            self._prompt_status(
                estimated_tokens
            )
        )

        result = PromptResult(
            scenario_id=scenario_id,
            prompt=user_prompt,
            model_input=model_input,
            character_count=len(user_prompt),
            estimated_tokens=estimated_tokens,
            status=status,
        )

        # --------------------------------------------------------
        # Logging.
        # --------------------------------------------------------

        if self.logger:
            self.logger.prompt_created(
                characters=len(user_prompt)
            )

        return result

    # ============================================================
    # SCENARIO CONTEXT
    # ============================================================

    def _build_scenario_context(
        self,
        scenario: Scenario,
    ) -> str:
        """
        Extract ONLY the neutral scenario context.

        This method deliberately does NOT fall back to:

            scenario.raw_markdown
            scenario.sections

        because those may contain ground truth.

        The preferred source is:

            scenario.neutral_sections

        If the updated ScenarioLoader stores a ScenarioContext
        object on the Scenario, that is also supported.
        """

        # --------------------------------------------------------
        # Preferred path:
        #
        # ScenarioContext attached to Scenario.
        # --------------------------------------------------------

        context = getattr(
            scenario,
            "context",
            None,
        )

        if isinstance(
            context,
            ScenarioContext,
        ):
            if not context.loaded:
                raise ValueError(
                    "ScenarioContext exists but is marked "
                    f"as not loaded for "
                    f"'{self._scenario_name(scenario)}'."
                )

            if not context.text.strip():
                raise ValueError(
                    "scenario.md was loaded but produced no "
                    "neutral model-visible context for "
                    f"'{self._scenario_name(scenario)}'."
                )

            return context.text.strip()

        # --------------------------------------------------------
        # Compatibility path:
        #
        # Older Scenario model with neutral_sections.
        #
        # This keeps the builder compatible during migration.
        # --------------------------------------------------------

        neutral_sections = getattr(
            scenario,
            "neutral_sections",
            None,
        )

        if neutral_sections:
            return self._render_neutral_sections(
                neutral_sections
            )

        # --------------------------------------------------------
        # If ScenarioLoader has not supplied neutral context,
        # FAIL rather than silently generating a source-only
        # prompt.
        #
        # This directly prevents the problem you had where
        # scenario.md existed in debug information but disappeared
        # from the actual model input.
        # --------------------------------------------------------

        raise ValueError(
            "No neutral scenario.md context is available for "
            f"'{self._scenario_name(scenario)}'. "
            "ScenarioLoader must load and provide "
            "neutral scenario context before PromptBuilder "
            "can construct the prompt."
        )

    # ============================================================
    # NEUTRAL SECTION RENDERING
    # ============================================================

    def _render_neutral_sections(
        self,
        neutral_sections,
    ) -> str:
        """
        Render neutral sections supplied by ScenarioLoader.

        Supports both:

            dict[str, str]

        and:

            list[ScenarioSection]
        """

        # --------------------------------------------------------
        # New ScenarioSection representation.
        # --------------------------------------------------------

        if isinstance(
            neutral_sections,
            list,
        ):
            blocks = []

            for section in neutral_sections:

                if not getattr(
                    section,
                    "included",
                    True,
                ):
                    continue

                heading = getattr(
                    section,
                    "heading",
                    "",
                ).strip()

                content = getattr(
                    section,
                    "content",
                    "",
                ).strip()

                if not content:
                    continue

                level = getattr(
                    section,
                    "level",
                    2,
                )

                level = max(
                    2,
                    min(
                        int(level),
                        6,
                    ),
                )

                blocks.append(
                    f"{'#' * level} {heading}\n\n"
                    f"{content}"
                )

            if not blocks:
                raise ValueError(
                    "ScenarioLoader supplied neutral sections, "
                    "but all sections were empty."
                )

            return "\n\n".join(
                blocks
            ).strip()

        # --------------------------------------------------------
        # Backward-compatible dictionary representation.
        # --------------------------------------------------------

        if isinstance(
            neutral_sections,
            dict,
        ):
            blocks = []

            for heading, content in (
                neutral_sections.items()
            ):
                if not content:
                    continue

                content = str(
                    content
                ).strip()

                if not content:
                    continue

                blocks.append(
                    f"## {str(heading).strip()}\n\n"
                    f"{content}"
                )

            if not blocks:
                raise ValueError(
                    "ScenarioLoader supplied neutral sections, "
                    "but all sections were empty."
                )

            return "\n\n".join(
                blocks
            ).strip()

        raise TypeError(
            "Unsupported neutral_sections type: "
            f"{type(neutral_sections).__name__}"
        )

    # ============================================================
    # SOURCE CODE
    # ============================================================

    def _build_source_code(
        self,
        code_files: list[CodeFile],
    ) -> str:
        """
        Build the model-visible source-code block.

        Only files already selected by PathResolver /
        CodeLoader are included.
        """

        blocks = []

        for code_file in code_files:

            content = (
                code_file.content.strip()
            )

            if not content:
                continue

            relative_path = (
                code_file.relative_path.strip()
            )

            if not relative_path:
                raise ValueError(
                    "A CodeFile has empty relative_path."
                )

            blocks.append(
                "## FILE: "
                f"{relative_path}\n\n"
                f"{content}"
            )

        if not blocks:
            raise ValueError(
                "All supplied source files are empty."
            )

        return "\n\n".join(
            blocks
        )

    # ============================================================
    # TEMPLATE RENDERING
    # ============================================================

    def _render_template(
        self,
        scenario_context: str,
        source_code: str,
    ) -> str:
        """
        Render prompt_template.txt.

        Supported placeholders:

            {{SCENARIO}}
            {{SOURCE_CODE}}

        Ground-truth placeholders are forbidden.
        """

        if not scenario_context.strip():
            raise ValueError(
                "Cannot render prompt without scenario context."
            )

        if not source_code.strip():
            raise ValueError(
                "Cannot render prompt without source code."
            )

        replacements = {
            "{{SCENARIO}}": scenario_context,
            "{{SOURCE_CODE}}": source_code,
        }

        rendered = self.prompt_template

        for placeholder, value in replacements.items():

            if placeholder not in rendered:
                raise ValueError(
                    "Prompt template is missing required "
                    f"placeholder: {placeholder}"
                )

            rendered = rendered.replace(
                placeholder,
                value,
            )

        rendered = rendered.strip()

        if not rendered:
            raise ValueError(
                "Rendered classification prompt is empty."
            )

        return rendered

    # ============================================================
    # TEMPLATE VALIDATION
    # ============================================================

    def _validate_template(self) -> None:
        """
        Validate prompt_template.txt before processing any
        scenarios.
        """

        required_placeholders = {
            "{{SCENARIO}}",
            "{{SOURCE_CODE}}",
        }

        missing = [
            placeholder
            for placeholder in required_placeholders
            if placeholder not in self.prompt_template
        ]

        if missing:
            raise ValueError(
                "prompt_template.txt is missing required "
                f"placeholder(s): {missing}"
            )

        # --------------------------------------------------------
        # No ground-truth placeholder may exist.
        # --------------------------------------------------------

        forbidden_placeholders = {
            "{{ATTACK}}",
            "{{TARGET}}",
            "{{LABEL}}",
            "{{CWE}}",
            "{{CVE}}",
            "{{CVE_DESC}}",
            "{{RED_REASONING}}",
            "{{BLUE_REASONING}}",
            "{{WRONG_ACTIONS}}",
            "{{WRONG_ACTION_REASONING}}",
            "{{OPTIMAL_ATTACK}}",
            "{{OPTIMAL_DEFENSE}}",
            "{{INJECTION_POINT}}",
            "{{SINK}}",
            "{{GROUND_TRUTH}}",
            "{{VULNERABILITY}}",
            "{{VULNERABILITY_TYPE}}",
            "{{SEVERITY}}",
            "{{VALID_ATTACKS}}",
            "{{VALID_DEFENSES}}",
            "{{COMPETING_HYPOTHESES}}",
            "{{CLASSIFICATION}}",
            "{{EXPECTED_CLASSIFICATION}}",
            "{{EXPECTED_ATTACK}}",
            "{{EXPECTED_DEFENSE}}",
        }

        found_forbidden = [
            placeholder
            for placeholder in forbidden_placeholders
            if placeholder in self.prompt_template
        ]

        if found_forbidden:
            raise ValueError(
                "Prompt template contains forbidden "
                "ground-truth placeholder(s): "
                f"{found_forbidden}"
            )

    # ============================================================
    # SCENARIO ID
    # ============================================================

    def _get_scenario_id(
        self,
        scenario: Scenario,
    ) -> str:
        """
        Resolve the scenario identifier.

        Prefer an explicit scenario_id supplied by the loader.

        IMPORTANT:
        Do not use the scenario name as the canonical ID.
        """

        scenario_id = getattr(
            scenario,
            "scenario_id",
            None,
        )

        if scenario_id:
            return str(
                scenario_id
            ).strip()

        context = getattr(
            scenario,
            "context",
            None,
        )

        if isinstance(
            context,
            ScenarioContext,
        ):
            if context.scenario_id:
                return context.scenario_id.strip()

        # Compatibility fallback.
        scenario_name = getattr(
            scenario,
            "scenario_name",
            None,
        )

        if scenario_name:
            return str(
                scenario_name
            ).strip()

        raise ValueError(
            "Unable to determine scenario ID."
        )

    # ============================================================
    # SCENARIO NAME
    # ============================================================

    @staticmethod
    def _scenario_name(
        scenario: Scenario,
    ) -> str:
        return str(
            getattr(
                scenario,
                "scenario_name",
                "unknown",
            )
        )

    # ============================================================
    # TOKEN ESTIMATION
    # ============================================================

    def _estimate_tokens(
        self,
        prompt: Prompt,
    ) -> int:
        """
        Rough token estimate.

        This is NOT an exact tokenizer count.

        A later tokenizer-based ContextEstimator can replace
        this value without changing PromptBuilder.
        """

        total_characters = (
            len(prompt.system)
            + len(prompt.user)
        )

        return max(
            1,
            total_characters // 4,
        )

    # ============================================================
    # PROMPT STATUS
    # ============================================================

    def _prompt_status(
        self,
        estimated_tokens: int,
    ) -> str:
        """
        Assign a coarse status based on estimated token count.

        These thresholds are intentionally conservative and
        should eventually come from config/context_estimator.
        """

        if estimated_tokens <= 4096:
            return "OK"

        if estimated_tokens <= 8192:
            return "WARNING"

        return "CRITICAL"

    # ============================================================
    # FILE LOADING
    # ============================================================

    def _load_template_file(
        self,
        path: Path,
        description: str,
    ) -> str:
        """
        Read system_prompt.txt or prompt_template.txt.
        """

        if not path.exists():
            raise FileNotFoundError(
                f"{description.capitalize()} file not found: "
                f"{path}"
            )

        if not path.is_file():
            raise ValueError(
                f"{description.capitalize()} path is not "
                f"a file: {path}"
            )

        try:

            content = path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError as exc:

            raise ValueError(
                f"Could not decode {description} as UTF-8: "
                f"{path}"
            ) from exc

        except OSError as exc:

            raise OSError(
                f"Could not read {description}: "
                f"{path}\n{exc}"
            ) from exc

        content = content.strip()

        if not content:
            raise ValueError(
                f"{description.capitalize()} is empty: "
                f"{path}"
            )

        return content

    # ============================================================
    # DEBUG PREVIEW
    # ============================================================

    def build_preview(
        self,
        prompt_result: PromptResult,
        max_characters: int = 5000,
    ) -> str:
        """
        Generate a debug preview of the exact prompt.

        This uses PromptResult.prompt, which is the same rendered
        prompt generated from ModelInput.
        """

        prompt_text = (
            prompt_result.prompt
        )

        preview = (
            "===== SYSTEM =====\n"
            f"{self.system_prompt}\n\n"
            "===== USER =====\n"
            f"{prompt_text}"
        )

        if len(preview) <= max_characters:
            return preview

        return (
            preview[:max_characters]
            + "\n\n...[TRUNCATED PREVIEW]..."
        )

    # ============================================================
    # PROMPT STATISTICS
    # ============================================================

    def prompt_statistics(
        self,
        prompt_result: PromptResult,
    ) -> dict:
        """
        Return statistics for debugging and build reports.
        """

        prompt = prompt_result.prompt

        model_input = (
            prompt_result.model_input
        )

        return {
            "scenario_id": (
                prompt_result.scenario_id
            ),

            "system_characters": len(
                self.system_prompt
            ),

            "user_characters": len(
                prompt
            ),

            "total_characters": (
                len(self.system_prompt)
                + len(prompt)
            ),

            "estimated_tokens": (
                prompt_result.estimated_tokens
            ),

            "status": (
                prompt_result.status
            ),

            "scenario_context_characters": (
                len(
                    model_input.scenario_context
                )
                if model_input
                else 0
            ),

            "source_code_characters": (
                len(
                    model_input.source_code
                )
                if model_input
                else 0
            ),

            "source_file_count": (
                len(
                    model_input.source_files
                )
                if model_input
                else 0
            ),

            "source_sections": (
                prompt.count(
                    "## FILE:"
                )
            ),

            "has_application_context": (
                "APPLICATION CONTEXT"
                in prompt
            ),

            "has_source_code": (
                "SOURCE CODE"
                in prompt
            ),
        }