"""
scenario_loader.py

Generic loader for scenario.md.

Supports:
- Markdown headings (#, ##, ### ...)
- Legacy "Field:" syntax
- Tables
- Fenced code blocks

The loader never decides what is safe to expose.
It simply parses the document into sections.

PromptBuilder decides which sections are included.
"""

from pathlib import Path

from builders.models import Scenario
from utils.file_utils import FileUtils

class ScenarioLoader:

    def __init__(self, logger):

        self.logger = logger

    # Public API

    def load(self, scenario_path: Path) -> Scenario:

        self.logger.title("Loading scenario.md")

        self.logger.info(f"Reading : {scenario_path}")

        if not scenario_path.exists():

            raise FileNotFoundError(f"\nscenario.md not found:\n{scenario_path}")

        markdown = FileUtils.read_text(scenario_path)

        sections = self._parse_sections(markdown)

        scenario = Scenario(
            scenario_name=scenario_path.parent.name,
            raw_markdown=markdown,
            sections=sections
        )

        self._print_summary(scenario)

        return scenario

    # Generic Markdown Parser

    def _parse_sections(self, markdown):

        sections = {}
        current_heading = None
        buffer = []
        inside_code_block = False

        for line in markdown.splitlines():

            stripped = line.strip()

            # Toggle fenced code block

            if stripped.startswith("```"):

                inside_code_block = not inside_code_block
                buffer.append(line)

                continue

            # Ignore heading detection inside code blocks

            if not inside_code_block:

                # Markdown Heading

                if stripped.startswith("#"):

                    self._save_section(
                        sections,
                        current_heading,
                        buffer
                    )

                    current_heading = stripped.lstrip("#").strip()

                    buffer = []

                    continue

                # Legacy Heading
                # Example:
                # Description:
                # Expected Attack:

                if (

                    stripped.endswith(":")

                    and "|" not in stripped

                    and len(stripped) < 80

                ):

                    self._save_section(
                        sections,
                        current_heading,
                        buffer
                    )

                    current_heading = stripped[:-1].strip()

                    buffer = []

                    continue

            buffer.append(line)

        self._save_section(
            sections,
            current_heading,
            buffer
        )

        return sections

    # Helpers

    def _save_section(self,sections,heading,buffer):

        if heading is None:

            return

        text = "\n".join(buffer).strip()

        if not text:

            return

        heading = " ".join(heading.split())

        sections[heading] = text

    # ==========================================================
    # Logging
    # ==========================================================

    def _print_summary(self, scenario):

        self.logger.success("Scenario Loaded Successfully")
        self.logger.info(f"Scenario : {scenario.scenario_name}")
        self.logger.info(f"Sections Found : {len(scenario.sections)}")

        for heading in scenario.sections:

            self.logger.info(f"  • {heading}")

        self.logger.info(f"Markdown Length : {len(scenario.raw_markdown)} characters")

# Reads scenario.md.
# Example:
# Scenario021/scenario.md

# Checks that scenario.md exists.
# Stops the builder if it is missing.
# Reads the markdown file.
# Loops through the markdown line by line.

# Finds section headings.
# Example:
# # Scenario
# ## Goal
# Description:

# Collects all text under each heading.

# Ignores headings inside code blocks.
# Example:
# ```python
# # Not a markdown heading
# ```

# Creates a dictionary of sections.
# Example:
# {
#   "Scenario": "...",
#   "Goal": "..."
# }

# Creates a Scenario object.
# Example:
# Scenario(
#   scenario_name="Scenario021",
#   sections={...}
# )

# Prints a summary of the loaded scenario.
# Returns the Scenario object to the PromptBuilder.