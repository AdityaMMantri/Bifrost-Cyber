from __future__ import annotations
from pathlib import Path
from .models import CodeFile, ResolvedFile

class CodeLoader:
    """
    Loads files already resolved by PathResolver.

    CodeLoader does NOT decide whether a file is code,
    relevant, or noise.

    PathResolver has already made that decision.

    Every ResolvedFile supplied here is loaded.
    """

    def __init__(self,logger=None,max_file_size: int | None = None):
        self.logger = logger
        self.max_file_size = max_file_size

    def load(self,resolved_files: list[ResolvedFile]) -> list[CodeFile]:
        """
        Load every resolved file.

        No extension filtering is performed.
        """

        return self.load_files(resolved_files)

    def load_files(self,resolved_files: list[ResolvedFile]) -> list[CodeFile]:
        """
        Load all files supplied by PathResolver.
        """

        if not resolved_files:
            raise ValueError("No resolved files were provided to CodeLoader.")

        code_files: list[CodeFile] = []

        for resolved_file in resolved_files:
            code_file = self.load_file(resolved_file)
            code_files.append(code_file)

        if self.logger:
            self.logger.info(f"Loaded {len(code_files)} source file(s)")

        return code_files

    # ============================================================
    # SINGLE FILE
    # ============================================================

    def load_file(self,resolved_file: ResolvedFile) -> CodeFile:
        """
        Load one resolved file.

        The file_type assigned by PathResolver is preserved.
        """

        if resolved_file is None:
            raise ValueError("ResolvedFile cannot be None.")
        path = Path(resolved_file.absolute_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Source path is not a file: {path}")

        # Optional size guard

        if self.max_file_size is not None:
            size = path.stat().st_size
            if size > self.max_file_size:
                raise ValueError(
                    f"Source file exceeds configured size limit: "
                    f"{path} "
                    f"({size:,} bytes > "
                    f"{self.max_file_size:,} bytes)"
                )

        # Read file
        content = self._read_text(path)
        # Statistics
        line_count = (len(content.splitlines()) if content else 0)
        character_count = len(content)
        file_type = resolved_file.file_type

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
                f"[{file_type}] "
                f"({line_count:,} lines, "
                f"{character_count:,} characters)"
            )

        return code_file

    # ============================================================
    # TEXT READING
    # ============================================================

    def _read_text(self,path: Path) -> str:
        """
        Read a codebase file as text.

        No extension-based decisions are made.
        """

        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            if self.logger:
                self.logger.warning(
                    f"UTF-8 decoding failed for {path}; "
                    f"retrying with latin-1."
                )
            try:
                return path.read_text(encoding="latin-1")
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
    # CODE FORMATTING
    # ============================================================

    def format_code_files(self,code_files: list[CodeFile]) -> str:
        """
        Combine all loaded files into the source-code block.

        Relevant and noise files are intentionally formatted
        identically. The model is not told which files are
        relevant and which are distractors.
        """

        if not code_files:
            raise ValueError("Cannot format an empty code-file list.")
        blocks: list[str] = []
        for code_file in code_files:
            blocks.append(
                f"## FILE: "
                f"{code_file.relative_path}\n\n"
                f"{code_file.content}"
            )

        return "\n\n".join(blocks)

    # ============================================================
    # STATISTICS
    # ============================================================

    def total_characters(self,code_files: list[CodeFile]) -> int:
        return sum(code_file.character_count for code_file in code_files)
    def total_lines(self,code_files: list[CodeFile]) -> int:
        return sum(code_file.line_count for code_file in code_files)
    def total_files(self,code_files: list[CodeFile]) -> int:
        return len(code_files)
    def total_relevant_files(self,code_files: list[CodeFile]) -> int:
        return sum(
            1
            for code_file in code_files
            if code_file.file_type == "relevant"
        )

    def total_noise_files(self,code_files: list[CodeFile]) -> int:
        return sum(
            1
            for code_file in code_files
            if code_file.file_type == "noise"
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    def validate_code_files(self,code_files: list[CodeFile]) -> None:
        """
        Validate loaded files.

        There is intentionally NO extension validation.
        """

        if not code_files:
            raise ValueError("No code files were loaded.")
        seen_paths: set[str] = set()
        for code_file in code_files:
            if not code_file.relative_path:
                raise ValueError("CodeFile has an empty relative path.")
            if not isinstance(code_file.content,str):
                raise ValueError(
                    f"CodeFile content must be a string: "
                    f"{code_file.relative_path}"
                )

            if not code_file.content.strip():
                raise ValueError(
                    f"Source file is empty: "
                    f"{code_file.relative_path}"
                )

            canonical = str(Path(code_file.absolute_path).resolve()).lower()
            if canonical in seen_paths:
                raise ValueError(
                    f"Duplicate source file loaded: "
                    f"{code_file.relative_path}"
                )

            seen_paths.add(canonical)

    # ============================================================
    # DEBUGGING
    # ============================================================

    def describe(self,code_files: list[CodeFile]) -> str:
        """
        Return a readable summary of loaded files.
        """

        if not code_files:
            return "No source files loaded."

        lines: list[str] = []

        for index, code_file in enumerate(code_files,start=1):
            lines.append(
                f"{index}. "
                f"[{code_file.file_type}] "
                f"{code_file.relative_path} | "
                f"{code_file.line_count:,} lines | "
                f"{code_file.character_count:,} chars"
            )

        lines.extend(
            [
                "",
                f"Total files    : "
                f"{self.total_files(code_files)}",
                f"Relevant files : "
                f"{self.total_relevant_files(code_files)}",
                f"Noise files    : "
                f"{self.total_noise_files(code_files)}",
                f"Total lines    : "
                f"{self.total_lines(code_files):,}",
                f"Total characters: "
                f"{self.total_characters(code_files):,}",
            ]
        )

        return "\n".join(lines)