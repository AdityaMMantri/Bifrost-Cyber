from __future__ import annotations

import logging
from pathlib import Path


class Logger:
    """
    Centralized logging utility for the PrimeVul dataset builder.

    Provides:
        - Console logging
        - File logging
        - Consistent formatting
        - Scenario-level progress reporting
        - Error and warning reporting

    The logger safely supports multiple Logger instances without
    duplicating handlers.
    """

    def __init__(
        self,
        name: str = "PrimeVulBuilder",
        log_file: Path | None = None,
        level: int = logging.INFO,
    ):
        self.logger = logging.getLogger(name)

        self.logger.setLevel(level)

        self.logger.propagate = False

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # ========================================================
        # CONSOLE HANDLER
        # ========================================================

        console_exists = any(
            isinstance(
                handler,
                logging.StreamHandler,
            )
            and not isinstance(
                handler,
                logging.FileHandler,
            )
            for handler in self.logger.handlers
        )

        if not console_exists:

            console_handler = logging.StreamHandler()

            console_handler.setLevel(level)

            console_handler.setFormatter(
                formatter
            )

            self.logger.addHandler(
                console_handler
            )

        # ========================================================
        # FILE HANDLER
        # ========================================================

        if log_file is not None:

            log_file = Path(log_file)

            log_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            resolved_log_file = (
                log_file.resolve()
            )

            file_exists = False

            for handler in self.logger.handlers:

                if not isinstance(
                    handler,
                    logging.FileHandler,
                ):
                    continue

                try:
                    existing_file = Path(
                        handler.baseFilename
                    ).resolve()

                    if (
                        existing_file
                        == resolved_log_file
                    ):
                        file_exists = True
                        break

                except Exception:
                    continue

            if not file_exists:

                file_handler = logging.FileHandler(
                    resolved_log_file,
                    encoding="utf-8",
                )

                file_handler.setLevel(level)

                file_handler.setFormatter(
                    formatter
                )

                self.logger.addHandler(
                    file_handler
                )

        # ========================================================
        # KEEP INSTANCE CONFIGURATION
        # ========================================================

        self.log_file = (
            Path(log_file)
            if log_file is not None
            else None
        )

    # ============================================================
    # STANDARD LOGGING METHODS
    # ============================================================

    def debug(
        self,
        message: str,
    ) -> None:
        self.logger.debug(message)

    def info(
        self,
        message: str,
    ) -> None:
        self.logger.info(message)

    def warning(
        self,
        message: str,
    ) -> None:
        self.logger.warning(message)

    def error(
        self,
        message: str,
    ) -> None:
        self.logger.error(message)

    def exception(
        self,
        message: str,
    ) -> None:
        self.logger.exception(message)

    # ============================================================
    # PRIMEVUL HELPERS
    # ============================================================

    def start_build(
        self,
        dataset_root: Path,
    ) -> None:

        self.info("=" * 70)
        self.info("PrimeVul Dataset Builder")
        self.info("=" * 70)
        self.info(
            f"Dataset root : {dataset_root}"
        )

        if self.log_file is not None:
            self.info(
                f"Log file     : {self.log_file}"
            )

    def start_scenario(
        self,
        scenario_name: str,
    ) -> None:

        self.info("-" * 70)
        self.info(
            f"Processing scenario: {scenario_name}"
        )
        self.info("-" * 70)

    def scenario_summary(
        self,
        scenario_name: str,
        relevant_files: int,
        target: int,
        cwe=None,
        code_characters: int | None = None,
    ) -> None:

        self.info(
            f"Scenario       : {scenario_name}"
        )

        self.info(
            f"Relevant files : {relevant_files}"
        )

        self.info(
            "Target         : "
            + (
                "VULNERABLE (1)"
                if target == 1
                else "SAFE (0)"
            )
        )

        if cwe:
            self.info(
                f"CWE            : {cwe}"
            )

        if code_characters is not None:
            self.info(
                f"Code characters: {code_characters}"
            )

    def files_loaded(
        self,
        count: int,
    ) -> None:

        self.info(
            f"Loaded {count} relevant source file(s)"
        )

    def prompt_created(
        self,
        characters: int | None = None,
    ) -> None:

        if characters is not None:

            self.info(
                "Classification prompt created "
                f"({characters:,} characters)"
            )

        else:

            self.info(
                "Classification prompt created"
            )

    def record_written(
        self,
        output_file: Path,
    ) -> None:

        self.info(
            f"JSONL record written: {output_file}"
        )

    def scenario_complete(
        self,
        scenario_name: str,
    ) -> None:

        self.info(
            f"Completed: {scenario_name}"
        )

    def build_complete(
        self,
        total_scenarios: int,
        successful: int,
        failed: int,
        output_file: Path,
    ) -> None:

        self.info("=" * 70)
        self.info(
            "PrimeVul dataset build complete"
        )
        self.info("=" * 70)

        self.info(
            f"Total scenarios : {total_scenarios}"
        )

        self.info(
            f"Successful      : {successful}"
        )

        self.info(
            f"Failed          : {failed}"
        )

        self.info(
            f"Output          : {output_file}"
        )

        if self.log_file is not None:
            self.info(
                f"Log file        : {self.log_file}"
            )

        self.info("=" * 70)