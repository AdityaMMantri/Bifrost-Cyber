"""
red_sft_loader.py

Loads red_sft.json and converts every training example
into the canonical SFTExample model.
"""

from pathlib import Path

from builders.models import SFTExample
from utils.file_utils import FileUtils

class RedSFTLoader:

    def __init__(self, logger):

        self.logger = logger

    # Public API

    def load(self, sft_path: Path):

        self.logger.title("Loading red_sft.json")
        self.logger.info(f"Reading : {sft_path}")

        if not sft_path.exists():

            raise FileNotFoundError(f"\nred_sft.json not found:\n{sft_path}")

        data = FileUtils.read_json(sft_path)

        if not isinstance(data, list):

            raise TypeError("red_sft.json must contain a JSON array.")

        examples = []

        for index, item in enumerate(data, start=1):

            examples.append(self._parse_example(item,index))

        self.logger.success(f"Loaded {len(examples)} training examples.")

        return examples

    # Parse One Example

    def _parse_example(self,item,index):

        if "messages" not in item:

            raise ValueError(f"Example R{index}: Missing 'messages' field.")

        messages = item["messages"]

        if not isinstance(messages, list):

            raise TypeError(f"Example R{index}: 'messages' must be a list.")

        if len(messages) <3:

            raise ValueError(f"Example R{index}: Expected at least 3 messages.")

        system=messages[0]
        user=messages[1]
        assistant=messages[2]

        if system.get("role") != "system":

            raise ValueError(f"Example R{index}: First message must be 'system'.")

        if user.get("role") != "user":

            raise ValueError(f"Example R{index}: Second message must be 'user'.")

        if assistant.get("role") != "assistant":

            raise ValueError(f"Example R{index}: Third message must be 'assistant'.")

        question = user.get(

            "content",

            ""

        ).strip()

        answer = assistant.get(

            "content",

            ""

        ).strip()

        # validation

        if not question:

            raise ValueError(f"Example R{index}: Empty question.")

        if not answer:

            raise ValueError(f"Example R{index}: Empty answer.")

        example = SFTExample(
            example_id=f"R{index}",
            question=question,
            answer=answer
        )

        self.logger.info(f"{example.example_id} loaded")

        self.logger.info(f"Question Length : {len(question)}")

        self.logger.info(f"Answer Length   : {len(answer)}")

        return example

# Reads red_sft.json.
# Example:
# [
#   {"messages":[...]},
#   {"messages":[...]}
# ]

# Checks that the file contains a list of training examples.

# Loops through each training example one by one.

# Checks that each example has a "messages" field.

# Checks the conversation format.
# Example:
# system -> user -> assistant

# Extracts the question from the user message.
# Example:
# "Find all vulnerabilities."

# Extracts the answer from the assistant message.
# Example:
# "The vulnerability is in Auth.py..."

# Creates an SFTExample object.
# Example:
# SFTExample(
#   example_id="R1",
#   question="Find all vulnerabilities.",
#   answer="The vulnerability is..."
# )

# Prints information about the loaded example.
# Example:
# R1 loaded
# Question Length : 120
# Answer Length : 850

# Returns a list of SFTExample objects to the Dataset Builder.