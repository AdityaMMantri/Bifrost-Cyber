from pathlib import Path

from .models import CodeFile, ResolvedFile


class CodeLoader:
    """
    Loads source-code contents from resolved files.

    Responsibilities:
        - Read resolved source files
        - Preserve their relative paths
        - Detect basic file type
        - Calculate line/character statistics
        - Return CodeFile objects

    The loader does not decide which files are relevant.
    That decision has already been made by metadata_loader.py
    and path_resolver.py.
    """

    # ============================================================
    # Supported source-code extensions
    # ============================================================

    SOURCE_EXTENSIONS = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".sh",
        ".bash",
        ".sql",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".vue",
        ".dart",
        ".ex",
        ".exs",
        ".erl",
        ".hrl",
    }

    def __init__(
        self,
        logger=None,
        max_file_size: int | None = None,
    ):
        """
        Parameters
        ----------
        logger:
            Optional Logger instance.

        max_file_size:
            Optional maximum file size in bytes.

            None means no artificial limit.

            For this benchmark, keeping it None is recommended
            initially so that source code is not silently
            truncated.
        """

        self.logger = logger
        self.max_file_size = max_file_size

    # ============================================================
    # PUBLIC API
    # ============================================================

    def load_files(
        self,
        resolved_files: list[ResolvedFile],
    ) -> list[CodeFile]:
        """
        Load all resolved files.

        Parameters
        ----------
        resolved_files:
            Files returned by PathResolver.

        Returns
        -------
        list[CodeFile]
            Loaded source-code objects.
        """

        if not resolved_files:
            raise ValueError(
                "No resolved files were provided to CodeLoader."
            )

        code_files = []

        for resolved_file in resolved_files:

            code_file = self.load_file(
                resolved_file
            )

            code_files.append(code_file)

        if self.logger:
            self.logger.files_loaded(
                len(code_files)
            )

        return code_files

    # ============================================================
    # SINGLE FILE
    # ============================================================

    def load_file(
        self,
        resolved_file: ResolvedFile,
    ) -> CodeFile:
        """
        Read one resolved source file.
        """

        path = resolved_file.absolute_path

        if not path.exists():
            raise FileNotFoundError(
                f"Source file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Source path is not a file: {path}"
            )

        # --------------------------------------------------------
        # Optional file-size guard
        # --------------------------------------------------------

        if self.max_file_size is not None:

            size = path.stat().st_size

            if size > self.max_file_size:
                raise ValueError(
                    f"Source file exceeds configured size limit: "
                    f"{path} "
                    f"({size:,} bytes > "
                    f"{self.max_file_size:,} bytes)"
                )

        # --------------------------------------------------------
        # Read source
        # --------------------------------------------------------

        content = self._read_text(path)

        # --------------------------------------------------------
        # Statistics
        # --------------------------------------------------------

        line_count = (
            len(content.splitlines())
            if content
            else 0
        )

        character_count = len(content)

        file_type = (
            resolved_file.file_type
            if resolved_file.file_type != "unknown"
            else self._detect_file_type(path)
        )

        code_file = CodeFile(
            relative_path=resolved_file.relative_path,
            absolute_path=path,
            file_type=file_type,
            content=content,
            line_count=line_count,
            character_count=character_count,
        )

        if self.logger:
            self.logger.debug(
                f"Loaded source file: "
                f"{resolved_file.relative_path} "
                f"({line_count:,} lines, "
                f"{character_count:,} characters)"
            )

        return code_file

    # ============================================================
    # TEXT READING
    # ============================================================

    def _read_text(
        self,
        path: Path,
    ) -> str:
        """
        Read a source file as UTF-8 text.

        A small fallback is provided for legacy files that may
        contain Windows/legacy encoded characters.
        """

        try:
            return path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            if self.logger:
                self.logger.warning(
                    f"UTF-8 decoding failed for {path}; "
                    f"retrying with latin-1."
                )

            try:
                return path.read_text(
                    encoding="latin-1"
                )

            except OSError as exc:
                raise OSError(
                    f"Could not read source file: "
                    f"{path}\n{exc}"
                ) from exc

        except OSError as exc:
            raise OSError(
                f"Could not read source file: "
                f"{path}\n{exc}"
            ) from exc

    # ============================================================
    # FILE TYPE
    # ============================================================

    def _detect_file_type(
        self,
        path: Path,
    ) -> str:
        """
        Determine whether a file is a known source-code type.

        Returns:
            "source"
            "unknown"
        """

        suffix = path.suffix.lower()

        if suffix in self.SOURCE_EXTENSIONS:
            return "source"

        return "unknown"

    # ============================================================
    # CODE FORMATTING
    # ============================================================

    def format_code_files(
        self,
        code_files: list[CodeFile],
    ) -> str:
        """
        Combine multiple CodeFile objects into the source-code
        block used by prompt_template.txt.

        Example:

            ## FILE: backend/routes/imports.js

            <source code>

            ## FILE: processor/report_loader.py

            <source code>
        """

        if not code_files:
            raise ValueError(
                "Cannot format an empty code-file list."
            )

        blocks = []

        for code_file in code_files:

            blocks.append(
                f"## FILE: {code_file.relative_path}\n\n"
                f"{code_file.content}"
            )

        return "\n\n".join(blocks)

    # ============================================================
    # STATISTICS
    # ============================================================

    def total_characters(
        self,
        code_files: list[CodeFile],
    ) -> int:
        """
        Return total source-code characters.
        """

        return sum(
            code_file.character_count
            for code_file in code_files
        )

    def total_lines(
        self,
        code_files: list[CodeFile],
    ) -> int:
        """
        Return total source-code lines.
        """

        return sum(
            code_file.line_count
            for code_file in code_files
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate_code_files(
        self,
        code_files: list[CodeFile],
    ) -> None:
        """
        Validate loaded code files before prompt construction.
        """

        if not code_files:
            raise ValueError(
                "No code files were loaded."
            )

        seen_paths = set()

        for code_file in code_files:

            if not code_file.content.strip():
                raise ValueError(
                    f"Source file is empty: "
                    f"{code_file.relative_path}"
                )

            canonical = str(
                code_file.absolute_path.resolve()
            ).lower()

            if canonical in seen_paths:
                raise ValueError(
                    f"Duplicate source file loaded: "
                    f"{code_file.relative_path}"
                )

            seen_paths.add(canonical)

    # ============================================================
    # DEBUGGING
    # ============================================================

    def describe(
        self,
        code_files: list[CodeFile],
    ) -> str:
        """
        Return a readable summary of loaded source files.
        """

        if not code_files:
            return "No source files loaded."

        lines = []

        for index, code_file in enumerate(
            code_files,
            start=1,
        ):
            lines.append(
                f"{index}. "
                f"{code_file.relative_path} | "
                f"{code_file.line_count:,} lines | "
                f"{code_file.character_count:,} chars"
            )

        return "\n".join(lines)