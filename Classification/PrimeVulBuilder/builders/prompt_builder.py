from pathlib import Path

from .models import (
    CodeFile,
    Prompt,
    Scenario,
)


class PromptBuilder:
    """
    Builds the final PrimeVul-style classification prompt.

    MODEL-VISIBLE:
        1. system_prompt.txt
        2. SAFE / NEUTRAL sections extracted from scenario.md
        3. relevant source-code files

    BUILDER-ONLY:
        - metadata.json
        - red_sft.json
        - blue_sft.json
        - attack labels
        - Red-Team reasoning
        - CWE
        - injection point
        - sink
        - expected attack
        - optimal attack
        - ground-truth labels

    Final structure:

        SYSTEM PROMPT
             +
        SCENARIO.MD CONTEXT
             +
        SOURCE CODE
             +
        CLASSIFICATION TASK
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
        # Load system prompt
        # --------------------------------------------------------

        self.system_prompt = (
            self._load_template_file(
                self.system_prompt_path,
                "system prompt",
            )
        )

        # --------------------------------------------------------
        # Load prompt template
        # --------------------------------------------------------

        self.prompt_template = (
            self._load_template_file(
                self.prompt_template_path,
                "prompt template",
            )
        )

        # --------------------------------------------------------
        # Validate template immediately.
        # --------------------------------------------------------

        self._validate_template()

    # ============================================================
    # PUBLIC API
    # ============================================================

    def build(
        self,
        scenario: Scenario,
        code_files: list[CodeFile],
    ) -> Prompt:
        """
        Build the final model-visible classification prompt.

        scenario:
            Parsed scenario.md.

            IMPORTANT:
            ScenarioLoader must already have populated
            scenario.neutral_sections.

        code_files:
            Relevant source files selected by metadata.json.

        Returns:
            Prompt(system=..., user=...)
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
        # 1. scenario.md
        #
        # Only neutral_sections are allowed.
        #
        # DO NOT use:
        #   scenario.raw_markdown
        #   scenario.sections
        #
        # because they may contain ground truth.
        # --------------------------------------------------------

        scenario_context = (
            self._build_scenario_context(
                scenario
            )
        )

        # --------------------------------------------------------
        # 2. Relevant source code
        # --------------------------------------------------------

        source_code = (
            self._build_source_code(
                code_files
            )
        )

        # --------------------------------------------------------
        # 3. Insert both into prompt_template.txt
        # --------------------------------------------------------

        user_prompt = (
            self._render_template(
                scenario_context=scenario_context,
                source_code=source_code,
            )
        )

        prompt = Prompt(
            system=self.system_prompt,
            user=user_prompt,
        )

        # --------------------------------------------------------
        # Logging
        # --------------------------------------------------------

        if self.logger:
            self.logger.prompt_created(
                characters=len(user_prompt)
            )

        return prompt

    # ============================================================
    # SCENARIO.MD CONTEXT
    # ============================================================

    def _build_scenario_context(
        self,
        scenario: Scenario,
    ) -> str:
        """
        Build the model-visible context extracted from
        scenario.md.

        IMPORTANT:

        ScenarioLoader is responsible for deciding which
        sections are neutral.

        PromptBuilder does NOT independently inspect metadata
        or ground truth.

        It simply consumes:

            scenario.neutral_sections

        Therefore the flow is:

            scenario.md
                ↓
            ScenarioLoader
                ↓
            neutral_sections
                ↓
            PromptBuilder
                ↓
            model prompt
        """

        # --------------------------------------------------------
        # Explicit validation.
        #
        # This makes it impossible to silently generate a
        # prompt without using scenario.md.
        # --------------------------------------------------------

        if not scenario.raw_markdown.strip():
            raise ValueError(
                "scenario.md was not loaded or is empty for "
                f"scenario '{scenario.scenario_name}'."
            )

        if not scenario.sections:
            raise ValueError(
                "scenario.md was loaded but no sections were "
                f"parsed for scenario '{scenario.scenario_name}'."
            )

        # --------------------------------------------------------
        # neutral_sections may legitimately be empty if a
        # scenario.md contains only excluded/ground-truth
        # sections.
        #
        # We do NOT fall back to scenario.sections because that
        # could leak the answer.
        # --------------------------------------------------------

        if not scenario.neutral_sections:
            return (
                "No additional neutral application context "
                "was provided by scenario.md."
            )

        blocks = []

        for heading, content in (
            scenario.neutral_sections.items()
        ):
            if not content:
                continue

            content = content.strip()

            if not content:
                continue

            blocks.append(
                f"## {heading.strip()}\n\n"
                f"{content}"
            )

        if not blocks:
            return (
                "No additional neutral application context "
                "was provided by scenario.md."
            )

        return "\n\n".join(
            blocks
        )

    # ============================================================
    # SOURCE CODE
    # ============================================================

    def _build_source_code(
        self,
        code_files: list[CodeFile],
    ) -> str:
        """
        Build the model-visible source-code section.

        Only files already selected by the upstream
        PathResolver/CodeLoader are included.

        PromptBuilder does NOT decide which files are relevant.
        """

        blocks = []

        for code_file in code_files:

            content = (
                code_file.content.strip()
            )

            if not content:
                continue

            blocks.append(
                "## FILE: "
                f"{code_file.relative_path}\n\n"
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

        No ground-truth placeholder is permitted.
        """

        replacements = {
            "{{SCENARIO}}": scenario_context,
            "{{SOURCE_CODE}}": source_code,
        }

        rendered = (
            self.prompt_template
        )

        for placeholder, value in (
            replacements.items()
        ):

            if placeholder not in rendered:
                raise ValueError(
                    "Prompt template is missing required "
                    f"placeholder: {placeholder}"
                )

            rendered = rendered.replace(
                placeholder,
                value,
            )

        return rendered.strip()

    # ============================================================
    # TEMPLATE VALIDATION
    # ============================================================

    def _validate_template(self) -> None:
        """
        Validate prompt_template.txt.

        The template MUST contain:

            {{SCENARIO}}
            {{SOURCE_CODE}}

        It MUST NOT contain placeholders that directly expose
        ground truth.
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
        # Ground-truth placeholders are forbidden.
        # --------------------------------------------------------

        forbidden_placeholders = {
            "{{ATTACK}}",
            "{{TARGET}}",
            "{{CWE}}",
            "{{CVE}}",
            "{{RED_REASONING}}",
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
    # DEBUGGING / PREVIEW
    # ============================================================

    def build_preview(
        self,
        prompt: Prompt,
        max_characters: int = 5000,
    ) -> str:
        """
        Return a shortened prompt preview.

        This is only for debugging/logging.
        """

        preview = (
            "===== SYSTEM =====\n"
            f"{prompt.system}\n\n"
            "===== USER =====\n"
            f"{prompt.user}"
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
        prompt: Prompt,
    ) -> dict:
        """
        Return basic prompt statistics.

        Character counts are used instead of pretending to
        represent exact tokenizer token counts.
        """

        return {
            "system_characters": len(
                prompt.system
            ),

            "user_characters": len(
                prompt.user
            ),

            "total_characters": (
                len(prompt.system)
                + len(prompt.user)
            ),

            "source_sections": (
                prompt.user.count(
                    "## FILE:"
                )
            ),

            "scenario_sections": (
                self._count_scenario_sections(
                    prompt.user
                )
            ),
        }

    # ============================================================
    # INTERNAL STATISTICS
    # ============================================================

    def _count_scenario_sections(
        self,
        user_prompt: str,
    ) -> int:
        """
        Approximate number of scenario context sections.

        This is only a debugging statistic.
        """

        if "APPLICATION CONTEXT" not in user_prompt:
            return 0

        if "SOURCE CODE" not in user_prompt:
            return 0

        context_part = user_prompt.split(
            "SOURCE CODE",
            1,
        )[0]

        return context_part.count(
            "## "
        )