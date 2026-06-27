"""
blue_sft_loader.py

Loads blue_sft.json and converts every training example
into the canonical SFTExample model.

Also performs strict validation of the Blue SFT format.

Every final assistant response MUST contain only:

Defense: ...

Reasoning:
...
"""

from pathlib import Path
import re

from builders.models import SFTExample
from utils.file_utils import FileUtils


class BlueSFTLoader:

    def __init__(self, logger):

        self.logger = logger
        self.current_scenario = None

    # =====================================================
    # Public API
    # =====================================================

    def load(self, sft_path: Path):

        self.current_scenario = sft_path.parent.name

        self.logger.title("Loading blue_sft.json")
        self.logger.info(f"Scenario : {self.current_scenario}")
        self.logger.info(f"Reading  : {sft_path}")

        if not sft_path.exists():

            self._fail(
                0,
                f"blue_sft.json not found:\n{sft_path}"
            )

        data = FileUtils.read_json(sft_path)

        if not isinstance(data, list):

            self._fail(
                0,
                "blue_sft.json must contain a JSON array."
            )

        examples = []

        for index, item in enumerate(data, start=1):

            examples.append(
                self._parse_example(item, index)
            )

        self.logger.success(
            f"Loaded {len(examples)} training examples."
        )

        return examples

    # =====================================================
    # Parse One Example
    # =====================================================

    def _parse_example(self, item, index):

        if "messages" not in item:

            self._fail(
                index,
                "Missing 'messages' field."
            )

        messages = item["messages"]

        if not isinstance(messages, list):

            self._fail(
                index,
                "'messages' must be a list."
            )

        if len(messages) < 3:

            self._fail(
                index,
                "Expected at least 3 messages."
            )

        # -------------------------------------------------
        # Validate first message
        # -------------------------------------------------

        if messages[0].get("role") != "system":

            self._fail(
                index,
                "First message must be 'system'."
            )

        # -------------------------------------------------
        # Validate alternating conversation
        # -------------------------------------------------

        expected = "user"

        for i, message in enumerate(messages[1:], start=1):

            role = message.get("role")

            if role != expected:

                self._fail(
                    index,
                    f"Invalid conversation at message {i+1}. "
                    f"Expected '{expected}', found '{role}'."
                )

            expected = "assistant" if expected == "user" else "user"

        # Conversation must always end with assistant

        if messages[-1].get("role") != "assistant":

            self._fail(
                index,
                "Conversation must end with an assistant message."
            )

        # Previous message must be user

        if messages[-2].get("role") != "user":

            self._fail(
                index,
                "Final assistant message must follow a user message."
            )

        # -------------------------------------------------
        # Use FINAL user/assistant pair
        # -------------------------------------------------

        question = messages[-2].get(

            "content",

            ""

        ).strip()

        answer = messages[-1].get(

            "content",

            ""

        ).strip()

        if not question:

            self._fail(
                index,
                "Empty final user question."
            )

        if not answer:

            self._fail(
                index,
                "Empty final assistant response."
            )

        self._validate_answer(
            answer,
            index
        )

        example = SFTExample(

            example_id=f"B{index}",

            question=question,

            answer=answer

        )

        self.logger.info(f"{example.example_id} loaded")
        self.logger.info(f"Question Length : {len(question)}")
        self.logger.info(f"Answer Length   : {len(answer)}")

        return example

    # =====================================================
    # Validate Assistant Response
    # =====================================================

    def _validate_answer(self, answer, index):

        if not answer.startswith("Defense:"):

            self._fail(
                index,
                "Response must begin with 'Defense:'."
            )

        if "\nReasoning:" not in answer:

            self._fail(
                index,
                "Missing 'Reasoning:' section."
            )

        allowed_headings = {

            "Defense",

            "Reasoning",

            "wrong_actions",

            "wrong_action_reasoning"

        }

        headings = re.findall(

    r"^([A-Za-z][A-Za-z ]+):\s*$",

    answer,

    flags=re.MULTILINE

)

        for heading in headings:

            if heading not in allowed_headings:

                self._fail(

                    index,

                    f"Unexpected heading '{heading}:'. "
                    "Only 'Defense:' and 'Reasoning:' are allowed."

                )

    # =====================================================
    # Common Error Handler
    # =====================================================

    def _fail(self, index, message):

        error = (

            "\n"

            + "=" * 80

            + f"\nScenario : {self.current_scenario}"

            + f"\nExample  : B{index}"

            + f"\nError    : {message}"

            + "\n"

            + "=" * 80

        )

        self.logger.error(error)

        raise ValueError(error)