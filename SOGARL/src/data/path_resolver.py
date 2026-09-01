"""
path_resolver.py

Resolves which codebase files should be given to SOGARL.

Locked SOGARL strategy
----------------------

For every scenario:

    ALL relevant files
            +
       2–5 noise files
            ↓
       ResolvedFile list
            ↓
       CodeLoader


Responsibilities
----------------

This module:

    - finds the scenario's files/ directory
    - resolves relevant files from metadata
    - resolves the metadata-defined noise-file pool
    - selects 2–5 noise files
    - creates ResolvedFile objects
    - preserves relative paths

This module does NOT:

    - read source-code contents
    - load metadata.json
    - load scenario.md
    - load red_sft.json
    - load blue_sft.json
    - inspect programming languages
    - filter by file extension
    - build prompts
    - perform Oracle evaluation
"""

from pathlib import Path
import random

from src.data.models import ResolvedFile


class PathResolver:

    def __init__(
        self,
        noise_min=2,
        noise_max=5,
        seed=None,
        logger=None,
    ):
        """
        Parameters
        ----------
        noise_min:
            Minimum number of noise files to select.

        noise_max:
            Maximum number of noise files to select.

        seed:
            Optional random seed.

        logger:
            Optional project logger.
        """

        if noise_min < 0:
            raise ValueError(
                "noise_min must be >= 0."
            )

        if noise_max < noise_min:
            raise ValueError(
                "noise_max must be >= noise_min."
            )

        self.noise_min = noise_min
        self.noise_max = noise_max
        self.logger = logger

        self.random = random.Random(seed)

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def resolve(
        self,
        scenario_path,
        relevant_files,
        noise_files=None,
    ):
        """
        Resolve:

            ALL relevant files
            +
            2–5 noise files

        Parameters
        ----------
        scenario_path:
            Path to the scenario directory.

        relevant_files:
            List of relevant file paths from metadata.json.

        noise_files:
            List of noise file paths from metadata.json.

            If None, no noise candidates are available.

        Returns
        -------
        List[ResolvedFile]
        """

        scenario_path = Path(
            scenario_path
        ).resolve()

        self._validate_scenario(
            scenario_path
        )

        codebase_path = (
            scenario_path / "files"
        )

        self._validate_codebase(
            codebase_path
        )

        if relevant_files is None:
            relevant_files = []

        if noise_files is None:
            noise_files = []

        if not isinstance(
            relevant_files,
            list,
        ):
            raise TypeError(
                "relevant_files must be a list."
            )

        if not isinstance(
            noise_files,
            list,
        ):
            raise TypeError(
                "noise_files must be a list."
            )

        # --------------------------------------------------------------
        # Resolve ALL relevant files
        # --------------------------------------------------------------

        relevant_resolved = []

        for relative_path in relevant_files:

            resolved = self._resolve_file(
                codebase_path=codebase_path,
                relative_path=relative_path,
                file_type="relevant",
            )

            relevant_resolved.append(
                resolved
            )

        # --------------------------------------------------------------
        # Resolve noise candidates
        # --------------------------------------------------------------

        noise_candidates = []

        for relative_path in noise_files:

            resolved = self._resolve_file(
                codebase_path=codebase_path,
                relative_path=relative_path,
                file_type="noise",
            )

            noise_candidates.append(
                resolved
            )

        # --------------------------------------------------------------
        # Remove accidental overlap
        #
        # A file should never be both relevant and noise.
        # --------------------------------------------------------------

        relevant_paths = {
            file.relative_path
            for file in relevant_resolved
        }

        noise_candidates = [
            file
            for file in noise_candidates
            if file.relative_path
            not in relevant_paths
        ]

        # --------------------------------------------------------------
        # Select 2–5 noise files
        # --------------------------------------------------------------

        noise_resolved = (
            self._select_noise_files(
                noise_candidates
            )
        )

        # --------------------------------------------------------------
        # Final resolved file list
        # --------------------------------------------------------------

        resolved_files = (
            relevant_resolved
            + noise_resolved
        )

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------

        if self.logger:

            self.logger.title(
                "Resolving Scenario Files"
            )

            self.logger.info(
                f"Relevant files : "
                f"{len(relevant_resolved)}"
            )

            self.logger.info(
                f"Noise files    : "
                f"{len(noise_resolved)}"
            )

            self.logger.success(
                f"Total files    : "
                f"{len(resolved_files)}"
            )

            for file in resolved_files:

                self.logger.info(
                    f"[{file.file_type.upper()}] "
                    f"{file.relative_path}"
                )

        return resolved_files

    # ==================================================================
    # RESOLVE ONE FILE
    # ==================================================================

    def _resolve_file(
        self,
        codebase_path,
        relative_path,
        file_type,
    ):
        """
        Resolve one metadata-defined relative path
        against the scenario's files/ directory.

        No source code is read here.
        """

        if not isinstance(
            relative_path,
            str,
        ):
            raise TypeError(
                "File path must be a string."
            )

        relative_path = (
            relative_path
            .replace("\\", "/")
            .strip()
        )

        if not relative_path:
            raise ValueError(
                "File path cannot be empty."
            )

        # --------------------------------------------------------------
        # Handle metadata paths that may already contain "files/"
        #
        # Example:
        #
        # metadata:
        #     files/services/Auth.py
        #
        # codebase:
        #     scenario_001/files/
        #
        # We don't want:
        #
        # scenario_001/files/files/services/Auth.py
        # --------------------------------------------------------------

        normalized_path = relative_path

        if normalized_path.startswith(
            "files/"
        ):
            normalized_path = (
                normalized_path[
                    len("files/"):
                ]
            )

        file_path = (
            codebase_path
            / normalized_path
        ).resolve()

        # --------------------------------------------------------------
        # Security boundary
        #
        # Metadata must not escape files/.
        # --------------------------------------------------------------

        try:

            file_path.relative_to(
                codebase_path
            )

        except ValueError as exc:

            raise ValueError(
                "Resolved file escapes the scenario "
                "codebase:\n"
                f"{relative_path}"
            ) from exc

        # --------------------------------------------------------------
        # File existence
        #
        # Windows is normally case-insensitive while Kaggle/Linux is
        # case-sensitive. The dataset contains a few filenames whose
        # capitalization differs between metadata.json and the actual
        # filesystem name.
        #
        # First try the exact path. If that fails, perform a
        # case-insensitive lookup using the complete relative path.
        # --------------------------------------------------------------

        if not file_path.is_file():

            target_parts = Path(
                normalized_path
            ).parts

            case_insensitive_matches = []

            for candidate in codebase_path.rglob("*"):

                if not candidate.is_file():
                    continue

                try:

                    candidate_parts = (
                        candidate
                        .relative_to(codebase_path)
                        .parts
                    )

                except ValueError:

                    continue

                if len(candidate_parts) != len(
                    target_parts
                ):
                    continue

                if all(
                    actual.lower() == expected.lower()
                    for actual, expected
                    in zip(
                        candidate_parts,
                        target_parts,
                    )
                ):

                    case_insensitive_matches.append(
                        candidate.resolve()
                    )

            # ----------------------------------------------------------
            # Exactly one match
            #
            # Use the actual filesystem path.
            # ----------------------------------------------------------

            if len(
                case_insensitive_matches
            ) == 1:

                file_path = (
                    case_insensitive_matches[0]
                )

            # ----------------------------------------------------------
            # Multiple matches
            #
            # Never silently choose between ambiguous files.
            # ----------------------------------------------------------

            elif len(
                case_insensitive_matches
            ) > 1:

                matches = "\n".join(
                    f"  - {match}"
                    for match in case_insensitive_matches
                )

                raise ValueError(
                    "Ambiguous case-insensitive file match:\n"
                    f"Metadata path : {relative_path}\n"
                    f"Matches:\n"
                    f"{matches}"
                )

            # ----------------------------------------------------------
            # No match
            # ----------------------------------------------------------

            else:

                raise FileNotFoundError(
                    "File listed in metadata does not exist:\n"
                    f"Metadata path : {relative_path}\n"
                    f"Resolved path : {file_path}"
                )

        # --------------------------------------------------------------
        # Final security boundary check
        #
        # Re-check after case-insensitive resolution because file_path
        # may now point to the actual filesystem path.
        # --------------------------------------------------------------

        try:

            file_path.relative_to(
                codebase_path
            )

        except ValueError as exc:

            raise ValueError(
                "Resolved file escapes the scenario "
                "codebase:\n"
                f"{relative_path}"
            ) from exc

        # --------------------------------------------------------------
        # Final file validation
        # --------------------------------------------------------------

        if not file_path.is_file():

            raise ValueError(
                f"Resolved path is not a file:\n"
                f"{file_path}"
            )

        return ResolvedFile(
            relative_path=(
                file_path
                .relative_to(codebase_path)
                .as_posix()
            ),
            absolute_path=file_path,
            file_type=file_type,
        )

    # ==================================================================
    # SELECT NOISE
    # ==================================================================

    def _select_noise_files(
        self,
        noise_candidates,
    ):
        """
        Randomly select between noise_min and noise_max
        files from the metadata-defined noise pool.

        If fewer files are available than noise_min,
        all available files are selected.

        If no noise files are available,
        an empty list is returned.
        """

        if not noise_candidates:
            return []

        maximum = min(
            self.noise_max,
            len(noise_candidates)
        )

        minimum = min(
            self.noise_min,
            maximum
        )

        number_of_noise_files = (
            self.random.randint(
                minimum,
                maximum
            )
        )

        return self.random.sample(
            noise_candidates,
            number_of_noise_files
        )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    @staticmethod
    def _validate_scenario(
        scenario_path,
    ):
        """
        Validate the scenario directory.
        """

        if not scenario_path.exists():

            raise FileNotFoundError(
                f"Scenario directory does not exist:\n"
                f"{scenario_path}"
            )

        if not scenario_path.is_dir():

            raise ValueError(
                f"Scenario path is not a directory:\n"
                f"{scenario_path}"
            )

    @staticmethod
    def _validate_codebase(
        codebase_path,
    ):
        """
        Validate the scenario's files/ directory.
        """

        if not codebase_path.exists():

            raise FileNotFoundError(
                f"Codebase directory not found:\n"
                f"{codebase_path}"
            )

        if not codebase_path.is_dir():

            raise ValueError(
                f"Codebase path is not a directory:\n"
                f"{codebase_path}"
            )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def resolve_scenario_files(
    scenario_path,
    relevant_files,
    noise_files=None,
    noise_min=2,
    noise_max=5,
    seed=None,
):
    """
    Convenience wrapper around PathResolver.

    Example
    -------

        resolved_files = resolve_scenario_files(
            scenario_path,
            metadata["relevant_files"],
            metadata["noise_files"],
        )
    """

    resolver = PathResolver(
        noise_min=noise_min,
        noise_max=noise_max,
        seed=seed,
    )

    return resolver.resolve(
        scenario_path=scenario_path,
        relevant_files=relevant_files,
        noise_files=noise_files,
    )