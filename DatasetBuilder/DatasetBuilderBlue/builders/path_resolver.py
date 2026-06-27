"""
path_resolver.py

Resolves file paths listed inside metadata.json into
absolute filesystem paths.

Supports both:

    files/security/Auth.py

and

    security/Auth.py

without requiring metadata changes.
"""

from pathlib import Path
from builders.models import ResolvedFile


class PathResolver:

    def __init__(self, logger):

        self.logger = logger

    # Public API

    def resolve_files(self,scenario_root: Path,relevant_files,noise_files):

        resolved = []
        self.logger.title("Resolving Source Files")

        # -------------------------
        # Relevant Files
        # -------------------------

        for path in relevant_files:

            resolved.append(

                self._resolve(

                    scenario_root,

                    path,

                    "relevant"

                )

            )

        # -------------------------
        # Noise Files
        # -------------------------

        for path in noise_files:

            resolved.append(

                self._resolve(

                    scenario_root,

                    path,

                    "noise"

                )

            )

        self.logger.success(f"Resolved {len(resolved)} files.")

        return resolved

    # ======================================================
    # Resolve One File
    # ======================================================

    def _resolve(self,scenario_root: Path,metadata_path: str,file_type: str):

        normalized = metadata_path.replace("\\", "/")
        normalized = normalized.lstrip("/")

        # -----------------------------------------
        # Possible Locations
        # -----------------------------------------

        candidates = [

            scenario_root / normalized,

            scenario_root / "files" / normalized,

            scenario_root / "files" / normalized.replace(

                "files/", "", 1

            )

        ]

        absolute_path = None

        for candidate in candidates:

            if candidate.exists():

                absolute_path = candidate.resolve()

                break

        if absolute_path is None:

            raise FileNotFoundError(

                f"\nUnable to resolve:\n"

                f"{metadata_path}\n\n"

                f"Scenario:\n"

                f"{scenario_root}"

            )

        relative = absolute_path.relative_to(

            scenario_root / "files"

        )

        self.logger.info(

            f"[{file_type.upper()}] {relative}"

        )

        return ResolvedFile(

            relative_path=str(relative).replace("\\", "/"),

            absolute_path=absolute_path,

            file_type=file_type

        )
# Gets the file paths from metadata.json.
# Example:
# "security/Auth.py"
# "README.md"

# Loops through all relevant and noise files.

# Cleans the file path.
# Example:
# "security\\Auth.py" -> "security/Auth.py"

# Tries different possible file locations.
# Example:
# Scenario021/security/Auth.py
# Scenario021/files/security/Auth.py

# Finds the file if it exists.
# Stops the builder if the file cannot be found.

# Creates a ResolvedFile object.
# Example:
# ResolvedFile(
#   relative_path="security/Auth.py",
#   absolute_path="D:/Dataset/Scenario021/files/security/Auth.py",
#   file_type="relevant"
# )

# Prints the resolved file.
# Example:
# [RELEVANT] security/Auth.py

# Returns all ResolvedFile objects to the CodeLoader.