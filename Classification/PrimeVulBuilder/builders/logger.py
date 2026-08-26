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
    """

    def __init__(
        self,
        name: str = "PrimeVulBuilder",
        log_file: Path | None = None,
        level: int = logging.INFO,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Prevent duplicate handlers if the Logger is initialized
        # more than once.
        self.logger.propagate = False

        if not self.logger.handlers:

            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # ----------------------------------------------------
            # Console handler
            # ----------------------------------------------------
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(console_handler)

            # ----------------------------------------------------
            # File handler
            # ----------------------------------------------------
            if log_file is not None:
                log_file = Path(log_file)
                log_file.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                file_handler = logging.FileHandler(
                    log_file,
                    encoding="utf-8",
                )

                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)

                self.logger.addHandler(file_handler)

    # ============================================================
    # Standard logging methods
    # ============================================================

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def exception(self, message: str):
        self.logger.exception(message)

    # ============================================================
    # PrimeVul-specific helpers
    # ============================================================

    def start_build(self, dataset_root: Path):
        self.info("=" * 70)
        self.info("PrimeVul Dataset Builder")
        self.info("=" * 70)
        self.info(f"Dataset root : {dataset_root}")

    def start_scenario(self, scenario_name: str):
        self.info("-" * 70)
        self.info(f"Processing scenario: {scenario_name}")
        self.info("-" * 70)

    def scenario_summary(
        self,
        scenario_name: str,
        relevant_files: int,
        target: int,
        cwe: str | None = None,
        code_characters: int | None = None,
    ):
        self.info(f"Scenario       : {scenario_name}")
        self.info(f"Relevant files : {relevant_files}")
        self.info(
            f"Target         : "
            f"{'VULNERABLE (1)' if target == 1 else 'SAFE (0)'}"
        )

        if cwe:
            self.info(f"CWE            : {cwe}")

        if code_characters is not None:
            self.info(f"Code characters: {code_characters}")

    def files_loaded(self, count: int):
        self.info(f"Loaded {count} relevant source file(s)")

    def prompt_created(self, characters: int | None = None):
        if characters is not None:
            self.info(
                f"Classification prompt created "
                f"({characters:,} characters)"
            )
        else:
            self.info("Classification prompt created")

    def record_written(self, output_file: Path):
        self.info(f"✓ JSONL record written: {output_file}")

    def scenario_complete(self, scenario_name: str):
        self.info(f"✓ Completed: {scenario_name}")

    def build_complete(
        self,
        total_scenarios: int,
        successful: int,
        failed: int,
        output_file: Path,
    ):
        self.info("=" * 70)
        self.info("PrimeVul dataset build complete")
        self.info("=" * 70)
        self.info(f"Total scenarios : {total_scenarios}")
        self.info(f"Successful      : {successful}")
        self.info(f"Failed          : {failed}")
        self.info(f"Output          : {output_file}")
        self.info("=" * 70)