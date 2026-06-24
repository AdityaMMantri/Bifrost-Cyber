"""
jsonl_writer.py

Writes training examples into train_red.jsonl.

Also creates debug prompt files that contain the exact
conversation shown to the model.
"""

import json
from pathlib import Path

from config import *


class JSONLWriter:

    def __init__(self):

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------

    def write_example(

        self,

        jsonl_path: Path,

        scenario_name: str,

        example_name: str,

        prompt,

        assistant

    ):

        training_example = {

            # ------------------------------------------------
            # Metadata (ignored during training but useful for
            # splitting, evaluation and analysis)
            # ------------------------------------------------

            "scenario_id": scenario_name,

            "example_id": example_name,

            # ------------------------------------------------

            "messages": [

                {

                    "role": "system",

                    "content": prompt.system

                },

                {

                    "role": "user",

                    "content": prompt.user

                },

                {

                    "role": "assistant",

                    "content": assistant

                }

            ]

        }

        with open(

            jsonl_path,

            "a",

            encoding="utf-8"

        ) as f:

            json.dump(

                training_example,

                f,

                ensure_ascii=False

            )

            f.write("\n")

        print(

            f"✓ JSONL Written : {scenario_name} ({example_name})"

        )

        if SAVE_DEBUG_PROMPTS:

            self._save_debug_prompt(

                scenario_name,

                example_name,

                prompt,

                assistant

            )

    # --------------------------------------------------------

    def _save_debug_prompt(

        self,

        scenario_name,

        example_name,

        prompt,

        assistant

    ):

        debug_file = (

            DEBUG_DIR /

            f"{scenario_name}_{example_name}.txt"

        )

        with open(

            debug_file,

            "w",

            encoding="utf-8"

        ) as f:

            f.write("=" * 80)

            f.write("\nSYSTEM\n")

            f.write("=" * 80)

            f.write("\n\n")

            f.write(prompt.system)

            f.write("\n\n")

            f.write("=" * 80)

            f.write("\nUSER\n")

            f.write("=" * 80)

            f.write("\n\n")

            f.write(prompt.user)

            f.write("\n\n")

            f.write("=" * 80)

            f.write("\nASSISTANT\n")

            f.write("=" * 80)

            f.write("\n\n")

            f.write(assistant)

        print(

            f"✓ Debug Prompt Saved : {debug_file.name}"

        )