"""
code_loader.py

Loads source code files selected for a scenario.

Input:
    List[ResolvedFile]

Output:
    List[CodeFile]
"""

from builders.models import CodeFile
from utils.file_utils import FileUtils

class CodeLoader:

    def __init__(self,logger):

        self.logger=logger

    # Public API

    def load(self, resolved_files):

        self.logger.title("Loading Source Code")
        loaded_files = []

        for resolved in resolved_files:

            loaded_files.append(self._load_file(resolved))

        self.logger.success(f"Loaded {len(loaded_files)} source files.")

        return loaded_files

    # Load One File

    def _load_file(self, resolved):

        text=FileUtils.read_text(resolved.absolute_path)
        code=CodeFile(
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

        self.logger.info(f"    Lines : {code.line_count}")
        self.logger.info(f"    Chars : {code.character_count}")

        return code

# Gets a list of files from the previous step (PathResolver).
# Example:
# [
#   ResolvedFile("security/Auth.py"),
#   ResolvedFile("database/UserDAO.java")
# ]

# Loops through each file one by one.
# Opens the file and reads its source code.
# Example:
# "class Auth:\n    def login(): ..."

# Creates a CodeFile object to store everything about the file.
# Example:
# CodeFile(
#     relative_path="security/Auth.py",
#     file_type="relevant",
#     content="class Auth: ...",
#     line_count=120,
#     character_count=3548
# )

# Prints information about the loaded file.
# Example:
# [RELEVANT] security/Auth.py
# Lines : 120
# Chars : 3548

# Repeats the same process for all files.
# Returns a list of CodeFile objects to the PromptBuilder.
# Example:
# [
#   CodeFile(Auth.py),
#   CodeFile(UserDAO.java)
# ]