"""
logger.py

Central logging utility for SOGARL.

Provides consistent console and file logging across the project.

Used by:
    - scenario_loader.py
    - metadata_loader.py
    - path_resolver.py
    - code_loader.py
    - generator.py
    - oracle.py
    - episode_manager.py
    - checkpoint.py
    - training scripts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


class SOGARLLogger:
    """Central project logger."""

    def __init__(
        self,
        name: str = "SOGARL",
        log_file: Optional[str | Path] = None,
        level: str = "INFO",
        console: bool = True,
    ) -> None:

        self.name = name
        self.log_file = (
            Path(log_file)
            if log_file is not None
            else None
        )

        self.logger = logging.getLogger(name)
        self.logger.setLevel(
            self._get_level(level)
        )

        self.logger.propagate = False

        self._configure(
            console=console
        )

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    @staticmethod
    def _get_level(
        level: str,
    ) -> int:
        """Convert a logging level name into a logging constant."""

        level = level.upper()

        if not hasattr(
            logging,
            level,
        ):
            raise ValueError(
                f"Invalid logging level: {level}"
            )

        return getattr(
            logging,
            level,
        )

    def _configure(
        self,
        console: bool,
    ) -> None:
        """Configure console and file handlers."""

        if self.logger.handlers:
            return

        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s | "
                "%(levelname)-8s | "
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if console:

            console_handler = (
                logging.StreamHandler()
            )

            console_handler.setFormatter(
                formatter
            )

            self.logger.addHandler(
                console_handler
            )

        if self.log_file is not None:

            self.log_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            file_handler = (
                logging.FileHandler(
                    self.log_file,
                    encoding="utf-8",
                )
            )

            file_handler.setFormatter(
                formatter
            )

            self.logger.addHandler(
                file_handler
            )

    # ========================================================================
    # STANDARD LOGGING
    # ========================================================================

    def debug(
        self,
        message: str,
    ) -> None:
        """Log debug information."""

        self.logger.debug(
            message
        )

    def info(
        self,
        message: str,
    ) -> None:
        """Log normal progress information."""

        self.logger.info(
            message
        )

    def warning(
        self,
        message: str,
    ) -> None:
        """Log a warning."""

        self.logger.warning(
            message
        )

    def error(
        self,
        message: str,
    ) -> None:
        """Log an error."""

        self.logger.error(
            message
        )

    def exception(
        self,
        message: str,
    ) -> None:
        """
        Log an exception including traceback.

        Should normally be called inside an except block.
        """

        self.logger.exception(
            message
        )

    # ========================================================================
    # PROJECT-SPECIFIC HELPERS
    # ========================================================================

    def success(
        self,
        message: str,
    ) -> None:
        """Log successful completion."""

        self.logger.info(
            f"[SUCCESS] {message}"
        )

    def title(
        self,
        message: str,
    ) -> None:
        """Print a visually distinct section heading."""

        separator = "=" * 70

        self.logger.info(
            separator
        )

        self.logger.info(
            message
        )

        self.logger.info(
            separator
        )

    def step(
        self,
        step_number: int,
        message: str,
    ) -> None:
        """Log a numbered pipeline step."""

        self.logger.info(
            f"[STEP {step_number}] {message}"
        )

    # ========================================================================
    # TRAINING HELPERS
    # ========================================================================

    def episode(
        self,
        epoch: int,
        episode: int,
        scenario_id: str,
    ) -> None:
        """Log the beginning of an episode."""

        self.logger.info(
            f"[EPISODE] "
            f"Epoch={epoch} | "
            f"Episode={episode} | "
            f"Scenario={scenario_id}"
        )

    def training(
        self,
        message: str,
    ) -> None:
        """Log training-related information."""

        self.logger.info(
            f"[TRAINING] {message}"
        )

    def validation(
        self,
        message: str,
    ) -> None:
        """Log validation-related information."""

        self.logger.info(
            f"[VALIDATION] {message}"
        )

    def testing(
        self,
        message: str,
    ) -> None:
        """Log testing-related information."""

        self.logger.info(
            f"[TEST] {message}"
        )

    # ========================================================================
    # MODEL HELPERS
    # ========================================================================

    def generation(
        self,
        turn: int,
        model: str,
        count: int = 1,
    ) -> None:
        """Log a generation operation."""

        self.logger.info(
            f"[GENERATION] "
            f"Turn={turn} | "
            f"Model={model} | "
            f"Count={count}"
        )

    # ========================================================================
    # REWARD / ORACLE HELPERS
    # ========================================================================

    def oracle(
        self,
        message: str,
    ) -> None:
        """Log Oracle-related information."""

        self.logger.info(
            f"[ORACLE] {message}"
        )

    def reward(
        self,
        message: str,
    ) -> None:
        """Log reward-related information."""

        self.logger.info(
            f"[REWARD] {message}"
        )

    # ========================================================================
    # CHECKPOINT HELPERS
    # ========================================================================

    def checkpoint(
        self,
        message: str,
    ) -> None:
        """Log checkpoint operations."""

        self.logger.info(
            f"[CHECKPOINT] {message}"
        )


def create_logger(
    log_file: Optional[str | Path] = None,
    level: str = "INFO",
    name: str = "SOGARL",
) -> SOGARLLogger:
    """
    Create the standard SOGARL logger.

    Example:

        logger = create_logger(
            log_file="outputs/logs/training.log"
        )
    """

    return SOGARLLogger(
        name=name,
        log_file=log_file,
        level=level,
    )