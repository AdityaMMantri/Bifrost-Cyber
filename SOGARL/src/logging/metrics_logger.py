"""
metrics_logger.py

Records SOGARL training, Red, Blue, validation,
and weakness-sampling metrics.

Does not:
    - calculate rewards
    - calculate GRPO advantages
    - update models
    - perform Oracle evaluation
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricsLogger:

    def __init__(
        self,
        output_dir,
        logger=None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.logger = logger

        self.training_metrics_path = (
            self.output_dir / "training_metrics.jsonl"
        )

        self.red_metrics_path = (
            self.output_dir / "red_metrics.csv"
        )

        self.blue_metrics_path = (
            self.output_dir / "blue_metrics.csv"
        )

        self.weakness_report_path = (
            self.output_dir / "weakness_report.json"
        )

        self._red_rows: List[Dict[str, Any]] = []
        self._blue_rows: List[Dict[str, Any]] = []

    # ==================================================================
    # TRAINING METRICS
    # ==================================================================

    def log_training(
        self,
        metrics: Dict[str, Any],
    ) -> None:

        self._write_jsonl(
            self.training_metrics_path,
            metrics,
        )

    def log_episode(
        self,
        episode_id: str,
        scenario_id: str,
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:

        record = {
            "episode_id": episode_id,
            "scenario_id": scenario_id,
            "epoch": epoch,
            **metrics,
        }

        self.log_training(record)

    # ==================================================================
    # RED METRICS
    # ==================================================================

    def log_red(
        self,
        metrics: Dict[str, Any],
    ) -> None:

        self._red_rows.append(
            dict(metrics)
        )

    # ==================================================================
    # BLUE METRICS
    # ==================================================================

    def log_blue(
        self,
        metrics: Dict[str, Any],
    ) -> None:

        self._blue_rows.append(
            dict(metrics)
        )

    # ==================================================================
    # FLUSH CSV METRICS
    # ==================================================================

    def flush_red(self) -> None:

        self._write_csv(
            self.red_metrics_path,
            self._red_rows,
        )

    def flush_blue(self) -> None:

        self._write_csv(
            self.blue_metrics_path,
            self._blue_rows,
        )

    def flush(self) -> None:

        self.flush_red()
        self.flush_blue()

    # ==================================================================
    # WEAKNESS REPORT
    # ==================================================================

    def save_weakness_report(
        self,
        report: Dict[str, Any],
    ) -> None:

        self._write_json(
            self.weakness_report_path,
            report,
        )

    # ==================================================================
    # GENERIC METRIC WRITER
    # ==================================================================

    def log(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
        **extra,
    ) -> None:

        record = {
            "metric": name,
            "value": value,
        }

        if step is not None:
            record["step"] = step

        record.update(extra)

        self.log_training(record)

    # ==================================================================
    # FILE HELPERS
    # ==================================================================

    @staticmethod
    def _write_jsonl(
        path: Path,
        record: Dict[str, Any],
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    @staticmethod
    def _write_json(
        path: Path,
        data: Dict[str, Any],
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    @staticmethod
    def _write_csv(
        path: Path,
        rows: List[Dict[str, Any]],
    ) -> None:

        if not rows:
            return

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fields = sorted(
            {
                key
                for row in rows
                for key in row.keys()
            }
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

    # ==================================================================
    # SUMMARY
    # ==================================================================

    def summary(self) -> Dict[str, Any]:

        return {
            "training_metrics_file": str(
                self.training_metrics_path
            ),
            "red_metrics_file": str(
                self.red_metrics_path
            ),
            "blue_metrics_file": str(
                self.blue_metrics_path
            ),
            "weakness_report_file": str(
                self.weakness_report_path
            ),
            "red_records": len(
                self._red_rows
            ),
            "blue_records": len(
                self._blue_rows
            ),
        }

    def close(self) -> None:
        self.flush()