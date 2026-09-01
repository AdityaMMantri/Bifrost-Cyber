"""
code_loader.py

Loads source-code files selected by PathResolver.

Responsibilities
----------------
PathResolver decides WHICH files are included.

CodeLoader only:
    - reads the selected files
    - creates CodeFile objects
    - returns them to the next stage

CodeLoader does NOT:
    - select files
    - filter by programming language
    - inspect file extensions
    - load metadata.json
    - load scenario.md
    - load red_sft.json
    - load blue_sft.json
    - build prompts
    - perform Oracle evaluation
    - perform RL/GRPO operations

Expected input
--------------
List[ResolvedFile]

Expected output
---------------
List[CodeFile]
"""

from src.data.models import CodeFile
from src.utils.file_utils import FileUtils


class CodeLoader:

    def __init__(self, logger=None):
        """
        Initialize the CodeLoader.

        Parameters
        ----------
        logger:
            Optional project logger.
        """

        self.logger = logger

    # ==================================================================
    # Public API
    # ==================================================================

    def load(self, resolved_files):
        """
        Load all files selected by PathResolver.

        PathResolver has already determined:
            - relevant files
            - noise files

        CodeLoader simply reads those files and creates CodeFile objects.

        Parameters
        ----------
        resolved_files : list[ResolvedFile]
            Files selected by PathResolver.

        Returns
        -------
        list[CodeFile]
            Loaded source-code files.
        """

        if resolved_files is None:
            raise ValueError(
                "resolved_files cannot be None."
            )

        if not isinstance(resolved_files, list):
            raise TypeError(
                "resolved_files must be a list."
            )

        if self.logger:
            self.logger.title(
                "Loading Source Code"
            )

        loaded_files = []

        for resolved_file in resolved_files:

            if resolved_file is None:
                continue

            loaded_files.append(
                self._load_file(
                    resolved_file
                )
            )

        if self.logger:
            self.logger.success(
                f"Loaded {len(loaded_files)} source files."
            )

        return loaded_files

    # ==================================================================
    # Load One File
    # ==================================================================

    def _load_file(self, resolved_file):
        """
        Read one selected source-code file.

        No extension filtering is performed.

        This allows scenarios to contain any programming language
        or text-based configuration/source file.
        """

        if not hasattr(
            resolved_file,
            "absolute_path"
        ):
            raise TypeError(
                "resolved_file must be a "
                "ResolvedFile object."
            )

        if not resolved_file.absolute_path.exists():
            raise FileNotFoundError(
                "Selected codebase file does not exist:\n"
                f"{resolved_file.absolute_path}"
            )

        if not resolved_file.absolute_path.is_file():
            raise ValueError(
                "Selected codebase path is not a file:\n"
                f"{resolved_file.absolute_path}"
            )

        # --------------------------------------------------------------
        # Read source code
        # --------------------------------------------------------------

        text = FileUtils.read_text(
            resolved_file.absolute_path
        )

        if text is None:
            raise ValueError(
                "FileUtils.read_text() returned None for:\n"
                f"{resolved_file.absolute_path}"
            )

        # --------------------------------------------------------------
        # Create CodeFile
        # --------------------------------------------------------------

        code = CodeFile(
            relative_path=(
                resolved_file.relative_path
            ),
            absolute_path=(
                resolved_file.absolute_path
            ),
            file_type=(
                resolved_file.file_type
            ),
            content=text,
            line_count=len(
                text.splitlines()
            ),
            character_count=len(text),
        )

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------

        if self.logger:

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