"""
logger.py

Simple console + file logger for the Dataset Builder.
"""

from pathlib import Path
from datetime import datetime


class Logger:

    def __init__(self, log_file: Path):

        self.log_file = log_file

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            self.log_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write("=" * 80 + "\n")
            f.write("Dataset Builder Log\n")
            f.write(f"Started : {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")

    def _log(self, level, message):

        timestamp = datetime.now().strftime("%H:%M:%S")

        line = f"[{timestamp}] [{level}] {message}"

        print(line)

        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(line + "\n")

    def info(self, message):
        self._log("INFO", message)

    def success(self, message):
        self._log("SUCCESS", message)

    def warning(self, message):
        self._log("WARNING", message)

    def error(self, message):
        self._log("ERROR", message)

    def title(self, title):

        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write("\n" + "=" * 80 + "\n")
            f.write(title + "\n")
            f.write("=" * 80 + "\n")