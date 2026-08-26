from pathlib import Path

from .models import Metadata, ResolvedFile


class PathResolver:
    """
    Resolves the file paths listed in metadata.json to actual
    files inside the corresponding scenario directory.

    Responsibilities:
        - Resolve metadata.relevant_files
        - Resolve metadata.noise_files when requested
        - Handle paths with or without the `files/` prefix
        - Prevent paths from escaping the scenario directory
        - Return ResolvedFile objects

    This class does NOT:
        - read file contents
        - modify metadata.json
        - decide whether a file is vulnerable
        - build prompts
    """

    def __init__(self, logger=None):
        self.logger = logger

    # ============================================================
    # PUBLIC API
    # ============================================================

    def resolve_relevant_files(
        self,
        scenario_path: Path,
        metadata: Metadata,
    ) -> list[ResolvedFile]:
        """
        Resolve only metadata.relevant_files.

        These files will eventually become the primary source-code
        input for the PrimeVul-style classification prompt.
        """

        return self.resolve_files(
            scenario_path=scenario_path,
            file_paths=metadata.relevant_files,
            file_type="relevant",
        )

    def resolve_noise_files(
        self,
        scenario_path: Path,
        metadata: Metadata,
    ) -> list[ResolvedFile]:
        """
        Resolve noise files.

        These are available for debugging/validation but should
        not be included in the PrimeVul model input.
        """

        return self.resolve_files(
            scenario_path=scenario_path,
            file_paths=metadata.noise_files,
            file_type="noise",
        )

    def resolve_files(
        self,
        scenario_path: Path,
        file_paths: list[str],
        file_type: str = "unknown",
    ) -> list[ResolvedFile]:
        """
        Resolve a list of metadata file paths.

        Handles both:

            files/backend/app.py

        and:

            backend/app.py

        when the scenario contains:

            scenario/files/backend/app.py
        """

        scenario_path = Path(
            scenario_path
        ).resolve()

        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Scenario directory not found: "
                f"{scenario_path}"
            )

        if not scenario_path.is_dir():
            raise ValueError(
                f"Scenario path is not a directory: "
                f"{scenario_path}"
            )

        resolved_files = []

        for relative_path in file_paths:

            resolved = self.resolve_single_file(
                scenario_path=scenario_path,
                relative_path=relative_path,
                file_type=file_type,
            )

            resolved_files.append(resolved)

        if self.logger:
            self.logger.info(
                f"Resolved {len(resolved_files)} "
                f"{file_type} file(s)"
            )

        return resolved_files

    # ============================================================
    # SINGLE FILE RESOLUTION
    # ============================================================

    def resolve_single_file(
        self,
        scenario_path: Path,
        relative_path: str,
        file_type: str = "unknown",
    ) -> ResolvedFile:
        """
        Resolve one metadata path to an actual file.

        The resolver tries the common layouts used by your
        scenarios:

            scenario/
                files/
                    backend/app.py

        Metadata:

            files/backend/app.py

        OR:

            backend/app.py

        Both resolve to:

            scenario/files/backend/app.py
        """

        scenario_path = Path(
            scenario_path
        ).resolve()

        if not isinstance(relative_path, str):
            raise TypeError(
                "Metadata file path must be a string, "
                f"got {type(relative_path).__name__}"
            )

        relative_path = relative_path.strip()

        if not relative_path:
            raise ValueError(
                "Cannot resolve an empty file path."
            )

        # Normalize Windows-style separators so that metadata
        # works consistently on Windows/Linux.
        normalized = relative_path.replace(
            "\\",
            "/",
        )

        # Remove a leading ./ if present.
        if normalized.startswith("./"):
            normalized = normalized[2:]

        # --------------------------------------------------------
        # Candidate 1:
        #
        # scenario/files/backend/app.py
        #
        # if metadata says:
        #
        # files/backend/app.py
        # --------------------------------------------------------

        candidate_paths = []

        direct_candidate = (
            scenario_path / normalized
        )

        candidate_paths.append(
            direct_candidate
        )

        # --------------------------------------------------------
        # Candidate 2:
        #
        # scenario/files/backend/app.py
        #
        # if metadata says:
        #
        # backend/app.py
        # --------------------------------------------------------

        if not normalized.startswith("files/"):
            candidate_paths.append(
                scenario_path
                / "files"
                / normalized
            )

        # --------------------------------------------------------
        # Candidate 3:
        #
        # Explicitly strip files/ and put it back under the
        # scenario's files directory.
        #
        # Useful for consistent handling of metadata paths.
        # --------------------------------------------------------

        if normalized.startswith("files/"):

            without_files = normalized[
                len("files/") :
            ]

            candidate_paths.append(
                scenario_path
                / "files"
                / without_files
            )

        # --------------------------------------------------------
        # Try candidates
        # --------------------------------------------------------

        for candidate in candidate_paths:

            try:
                resolved_candidate = (
                    candidate.resolve()
                )
            except OSError:
                continue

            # Security check:
            # the resolved file must remain inside the scenario.
            if not self._is_inside(
                resolved_candidate,
                scenario_path,
            ):
                raise ValueError(
                    f"Path escapes scenario directory: "
                    f"{relative_path}"
                )

            if resolved_candidate.is_file():

                canonical_relative = (
                    self._relative_display_path(
                        resolved_candidate,
                        scenario_path,
                    )
                )

                return ResolvedFile(
                    relative_path=canonical_relative,
                    absolute_path=resolved_candidate,
                    file_type=file_type,
                )

        # --------------------------------------------------------
        # File could not be found.
        # --------------------------------------------------------

        attempted = "\n".join(
            f"  - {path}"
            for path in candidate_paths
        )

        raise FileNotFoundError(
            f"Could not resolve metadata file:\n"
            f"  {relative_path}\n\n"
            f"Attempted:\n{attempted}"
        )

    # ============================================================
    # PATH SAFETY
    # ============================================================

    def _is_inside(
        self,
        path: Path,
        parent: Path,
    ) -> bool:
        """
        Ensure `path` is inside `parent`.

        This prevents metadata paths such as:

            ../../some_secret_file

        from escaping the scenario directory.
        """

        try:
            path.relative_to(parent)
            return True

        except ValueError:
            return False

    # ============================================================
    # DISPLAY PATH
    # ============================================================

    def _relative_display_path(
        self,
        path: Path,
        scenario_path: Path,
    ) -> str:
        """
        Produce a stable path for the generated dataset.

        Prefer:

            files/backend/app.py

        instead of an absolute Windows path.
        """

        relative = path.relative_to(
            scenario_path
        )

        return relative.as_posix()

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate_files(
        self,
        resolved_files: list[ResolvedFile],
    ) -> None:
        """
        Validate a collection of resolved files before they are
        passed to the code loader.
        """

        if not resolved_files:
            raise ValueError(
                "No files were resolved."
            )

        seen_paths = set()

        for resolved_file in resolved_files:

            path = resolved_file.absolute_path

            if not path.exists():
                raise FileNotFoundError(
                    f"Resolved file no longer exists: "
                    f"{path}"
                )

            if not path.is_file():
                raise ValueError(
                    f"Resolved path is not a file: "
                    f"{path}"
                )

            canonical = str(
                path.resolve()
            ).lower()

            if canonical in seen_paths:
                raise ValueError(
                    f"Duplicate resolved file: "
                    f"{path}"
                )

            seen_paths.add(canonical)

    # ============================================================
    # DEBUGGING
    # ============================================================

    def describe(
        self,
        resolved_files: list[ResolvedFile],
    ) -> str:
        """
        Return a human-readable representation of resolved files.
        """

        if not resolved_files:
            return "No files resolved."

        lines = []

        for index, file in enumerate(
            resolved_files,
            start=1,
        ):
            lines.append(
                f"{index}. "
                f"[{file.file_type}] "
                f"{file.relative_path}"
            )

        return "\n".join(lines)