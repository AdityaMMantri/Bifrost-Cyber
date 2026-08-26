import json
from pathlib import Path

from .models import PrimeVulRecord


class JSONLWriter:
    """
    Writes PrimeVulRecord objects to a JSON Lines (.jsonl) file.

    Each line represents exactly one dataset example.

    Responsibilities:
        - Validate PrimeVulRecord objects
        - Convert records to dictionaries
        - Serialize them as JSON
        - Write one record per line
        - Preserve deterministic field ordering
        - Prevent accidental malformed output

    This class does NOT:
        - determine vulnerability labels
        - load metadata
        - load Red SFT
        - build prompts
        - read source files
    """

    # ============================================================
    # PRIMEVUL OUTPUT FIELDS
    # ============================================================

    FIELD_ORDER = [
        "project",
        "commit_id",
        "target",
        "func",
        "cwe",
        "cve",
        "cve_desc",
    ]

    def __init__(
        self,
        output_path: Path,
        logger=None,
        overwrite: bool = True,
    ):
        """
        Parameters
        ----------
        output_path:
            Destination .jsonl file.

        logger:
            Optional Logger instance.

        overwrite:
            If True, an existing output file is replaced.
            If False, an existing output file raises an error.
        """

        self.output_path = Path(output_path)
        self.logger = logger
        self.overwrite = overwrite

    # ============================================================
    # PUBLIC API
    # ============================================================

    def write(
        self,
        records: list[PrimeVulRecord],
    ) -> Path:
        """
        Write all records to the JSONL output file.

        Each record occupies exactly one line.
        """

        if not records:
            raise ValueError(
                "Cannot write an empty JSONL dataset."
            )

        self._prepare_output_path()

        validated_records = []

        for index, record in enumerate(
            records,
            start=1,
        ):
            self._validate_record(
                record,
                index,
            )

            validated_records.append(
                self._record_to_dict(record)
            )

        try:
            with self.output_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as file:

                for record in validated_records:

                    json.dump(
                        record,
                        file,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )

                    file.write("\n")

        except OSError as exc:
            raise OSError(
                f"Could not write JSONL file: "
                f"{self.output_path}\n{exc}"
            ) from exc

        if self.logger:
            self.logger.info(
                f"Wrote {len(validated_records)} "
                f"records to {self.output_path}"
            )

        return self.output_path

    # ============================================================
    # SINGLE RECORD
    # ============================================================

    def write_record(
        self,
        record: PrimeVulRecord,
    ) -> Path:
        """
        Write one PrimeVulRecord to the output file.

        Useful when the main builder wants to stream records
        rather than keeping the entire dataset in memory.

        NOTE:
        This method appends to the file.
        """

        self._validate_record(
            record,
            1,
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        record_dict = self._record_to_dict(
            record
        )

        try:
            with self.output_path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as file:

                json.dump(
                    record_dict,
                    file,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                file.write("\n")

        except OSError as exc:
            raise OSError(
                f"Could not append to JSONL file: "
                f"{self.output_path}\n{exc}"
            ) from exc

        return self.output_path

    # ============================================================
    # RECORD CONVERSION
    # ============================================================

    def _record_to_dict(
        self,
        record: PrimeVulRecord,
    ) -> dict:
        """
        Convert PrimeVulRecord to the final JSONL schema.

        Internal/debug-only fields such as scenario_id are
        deliberately excluded.
        """

        return {
            "project": record.project,
            "commit_id": record.commit_id,
            "target": record.target,
            "func": record.func,
            "cwe": record.cwe,
            "cve": record.cve,
            "cve_desc": record.cve_desc,
        }

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_record(
        self,
        record: PrimeVulRecord,
        index: int,
    ) -> None:
        """
        Validate a record before it reaches the JSONL file.
        """

        if not isinstance(
            record,
            PrimeVulRecord,
        ):
            raise TypeError(
                f"Record {index} is not a "
                f"PrimeVulRecord object."
            )

        # --------------------------------------------------------
        # Project
        # --------------------------------------------------------

        if not isinstance(
            record.project,
            str,
        ) or not record.project.strip():

            raise ValueError(
                f"Record {index}: "
                f"'project' must be a non-empty string."
            )

        # --------------------------------------------------------
        # Target
        # --------------------------------------------------------

        if record.target not in (0, 1):

            raise ValueError(
                f"Record {index}: "
                f"'target' must be 0 or 1, "
                f"got {record.target!r}."
            )

        # --------------------------------------------------------
        # Source code
        # --------------------------------------------------------

        if not isinstance(
            record.func,
            str,
        ) or not record.func.strip():

            raise ValueError(
                f"Record {index}: "
                f"'func' must contain source code."
            )

        # --------------------------------------------------------
        # Optional fields
        # --------------------------------------------------------

        optional_fields = {
            "commit_id": record.commit_id,
            "cwe": record.cwe,
            "cve": record.cve,
            "cve_desc": record.cve_desc,
        }

        for field_name, value in (
            optional_fields.items()
        ):

            if value is not None and not isinstance(
                value,
                str,
            ):
                raise ValueError(
                    f"Record {index}: "
                    f"'{field_name}' must be a string "
                    f"or null."
                )

    # ============================================================
    # OUTPUT PREPARATION
    # ============================================================

    def _prepare_output_path(self) -> None:
        """
        Create the output directory and enforce overwrite rules.
        """

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            self.output_path.exists()
            and not self.overwrite
        ):
            raise FileExistsError(
                f"Output file already exists: "
                f"{self.output_path}"
            )

    # ============================================================
    # VALIDATION OF EXISTING JSONL
    # ============================================================

    def validate_existing_file(
        self,
        path: Path | None = None,
    ) -> dict:
        """
        Validate an already-created JSONL file.

        Returns summary statistics.

        This is useful after the build to catch malformed lines,
        invalid targets, or missing required fields.
        """

        path = (
            Path(path)
            if path is not None
            else self.output_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"JSONL file not found: {path}"
            )

        total = 0
        vulnerable = 0
        safe = 0
        errors = []

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                for line_number, line in enumerate(
                    file,
                    start=1,
                ):

                    line = line.strip()

                    if not line:
                        continue

                    total += 1

                    try:
                        record = json.loads(
                            line
                        )

                        self._validate_dict_record(
                            record,
                            line_number,
                        )

                        if record["target"] == 1:
                            vulnerable += 1
                        else:
                            safe += 1

                    except Exception as exc:
                        errors.append(
                            f"Line {line_number}: {exc}"
                        )

        except OSError as exc:
            raise OSError(
                f"Could not read JSONL file: "
                f"{path}\n{exc}"
            ) from exc

        return {
            "total_records": total,
            "vulnerable": vulnerable,
            "safe": safe,
            "errors": errors,
            "valid": len(errors) == 0,
        }

    # ============================================================
    # DICTIONARY VALIDATION
    # ============================================================

    def _validate_dict_record(
        self,
        record: dict,
        line_number: int,
    ) -> None:
        """
        Validate a raw JSON object from an existing JSONL file.
        """

        if not isinstance(record, dict):
            raise ValueError(
                "record is not a JSON object"
            )

        required_fields = [
            "project",
            "commit_id",
            "target",
            "func",
            "cwe",
            "cve",
            "cve_desc",
        ]

        missing = [
            field
            for field in required_fields
            if field not in record
        ]

        if missing:
            raise ValueError(
                f"missing fields: {missing}"
            )

        if record["target"] not in (0, 1):
            raise ValueError(
                f"invalid target: "
                f"{record['target']!r}"
            )

        if not isinstance(
            record["project"],
            str,
        ):
            raise ValueError(
                "'project' must be a string"
            )

        if not isinstance(
            record["func"],
            str,
        ) or not record["func"].strip():
            raise ValueError(
                "'func' is empty"
            )

    # ============================================================
    # STATISTICS
    # ============================================================

    def count_records(
        self,
        path: Path | None = None,
    ) -> int:
        """
        Count non-empty JSONL records.
        """

        path = (
            Path(path)
            if path is not None
            else self.output_path
        )

        if not path.exists():
            return 0

        count = 0

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                if line.strip():
                    count += 1

        return count