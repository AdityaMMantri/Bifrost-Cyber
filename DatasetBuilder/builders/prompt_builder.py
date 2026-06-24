"""
prompt_builder.py

Constructs the final prompt shown to the model.

Responsibilities
----------------
• Choose a random prompt template
• Format scenario context
• Format source code
• Replace template placeholders
"""

from pathlib import Path

from config import SAFE_SCENARIO_SECTIONS

from builders.models import Prompt
from utils.file_utils import FileUtils
from utils.random_utils import RandomUtils


class PromptBuilder:

    def __init__(

        self,

        system_prompt_path: Path,

        template_directory: Path,

        logger

    ):

        self.logger = logger

        self.random = RandomUtils()

        self.system_prompt = FileUtils.read_text(
            system_prompt_path
        )

        self.templates = sorted(
            template_directory.glob("*.txt")
        )

        if not self.templates:

            raise RuntimeError(
                "No prompt templates found."
            )

    # =====================================================
    # Build Prompt
    # =====================================================

    def build(

        self,

        scenario,

        code_files,

        question

    ) -> Prompt:

        self.logger.title("Building Prompt")

        template_path = self.random.choice(
            self.templates
        )

        self.logger.info(
            f"Template : {template_path.name}"
        )

        template = FileUtils.read_text(
            template_path
        )

        scenario_text = self._build_scenario_text(
            scenario
        )

        source_code = self._build_source_code(
            code_files
        )

        prompt = template

        prompt = prompt.replace(
            "{{SCENARIO}}",
            scenario_text
        )

        prompt = prompt.replace(
            "{{SOURCE_CODE}}",
            source_code
        )

        prompt = prompt.replace(
            "{{QUESTION}}",
            question
        )

        self.logger.success(
            "Prompt Constructed"
        )

        self.logger.info(
            f"Characters : {len(prompt)}"
        )

        self.logger.info(
            f"Files Included : {len(code_files)}"
        )

        return Prompt(

            system=self.system_prompt,

            user=prompt

        )

    # =====================================================
    # Scenario Formatting
    # =====================================================

    def _build_scenario_text(

        self,

        scenario

    ):

        output = []
        output.append(f"Application ID: {scenario.scenario_name}")
        output.append("")

        for heading, content in scenario.sections.items():

            if heading not in SAFE_SCENARIO_SECTIONS:
                continue

            output.append(f"## {heading}")
            output.append("")
            output.append(content.strip())
            output.append("")

        return "\n".join(output)

    # =====================================================
    # Source Code Formatting
    # =====================================================

    def _build_source_code(

        self,

        code_files

    ):

        relevant = [

            file

            for file in code_files

            if file.file_type == "relevant"

        ]

        noise = [

            file

            for file in code_files

            if file.file_type == "noise"

        ]

        ordered = relevant + noise

        output = []

        for file in ordered:

            output.append(
                f"## FILE: {file.relative_path}"
            )

            output.append("")

            output.append("```")

            output.append(file.content.rstrip())

            output.append("```")

            output.append("")

        return "\n".join(output)