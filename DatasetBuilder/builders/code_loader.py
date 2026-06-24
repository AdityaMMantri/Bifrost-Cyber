"""
code_loader.py

Loads source code files selected for a scenario.

Input:
    List[ResolvedFile]

Output:
    List[CodeFile]
"""

from pathlib import Path

from builders.models import CodeFile
from utils.file_utils import FileUtils


class CodeLoader:

    def __init__(self, logger):

        self.logger = logger

    # ======================================================
    # Public API
    # ======================================================

    def load(self, resolved_files):

        self.logger.title("Loading Source Code")

        loaded_files = []

        for resolved in resolved_files:

            loaded_files.append(self._load_file(resolved))

        self.logger.success(f"Loaded {len(loaded_files)} source files.")

        return loaded_files

    # ======================================================
    # Load One File
    # ======================================================

    def _load_file(self, resolved):

        text = FileUtils.read_text(

            resolved.absolute_path

        )

        code = CodeFile(

            relative_path=resolved.relative_path,

            absolute_path=resolved.absolute_path,

            file_type=resolved.file_type,

            content=text,

            line_count=len(text.splitlines()),

            character_count=len(text)

        )

        self.logger.info(

            f"[{code.file_type.upper()}] "

            f"{code.relative_path}"

        )

        self.logger.info(

            f"    Lines : {code.line_count}"

        )

        self.logger.info(

            f"    Chars : {code.character_count}"

        )

        return code