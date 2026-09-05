from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from .models import PrimeVulRecord


class JSONLWriter:
    """
    Writes PrimeVulRecord objects to a JSON Lines (.jsonl) file.

    PrimeVul-style output:

        {
            "project": ...,
            "commit_id": ...,
            "target": 0/1,
            "func": "...model-visible context + source...",
            "cwe": ...,
            "cve": ...,
            "cve_desc": ...
        }

    IMPORTANT
    ---------
    This writer does NOT construct model context.

    The model-visible `func` must already have been constructed
    by the builder/prompt pipeline.

    Therefore this class:

        - validates records
        - serializes records
        - preserves func exactly
        - never inserts metadata
        - never inserts attack labels
        - never inserts ground truth
        - never inserts Red SFT answers
        - never inserts scenario metadata

    Internal fields such as scenario_id and scenario_context
    are intentionally NOT serialized.
    """

    # ============================================================
    # PRIMEVUL OUTPUT SCHEMA
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

    REQUIRED_FIELDS = {
        "project",
        "commit_id",
        "target",
        "func",
        "cwe",
        "cve",
        "cve_desc",
    }

    # ============================================================
    # CONSTRUCTOR
    # ============================================================

    def __init__(self,output_path: Path,logger=None,overwrite: bool = True):
        self.output_path = Path(output_path)
        self.logger = logger
        self.overwrite = overwrite

    def write(self,records: list[PrimeVulRecord]) -> Path:
        """
        Write all records to JSONL.

        One record = one JSON line.
        """

        if not records:
            raise ValueError("Cannot write an empty JSONL dataset.")
        self._prepare_output_path()
        validated_records: list[dict[str, Any]] = []
        for index, record in enumerate(records,start=1):
            self._validate_record(record,index)
            validated_records.append(self._record_to_dict(record))
        try:

            with self.output_path.open("w",encoding="utf-8",newline="\n") as file:
                for record in validated_records:
                    json.dump(record,file,ensure_ascii=False,separators=(",",":",))
                    file.write("\n")

        except OSError as exc:
            raise OSError(f"Could not write JSONL file: "f"{self.output_path}\n{exc}") from exc

        if self.logger:
            self.logger.info(
                f"Wrote "
                f"{len(validated_records)} "
                f"records to "
                f"{self.output_path}"
            )

        return self.output_path

    # ============================================================
    # SINGLE RECORD
    # ============================================================

    def write_record(self,record: PrimeVulRecord) -> Path:
        """
        Append one record to the JSONL file.

        This does not alter the record.
        """

        self._validate_record(record,1)
        self.output_path.parent.mkdir(parents=True,exist_ok=True)
        record_dict = self._record_to_dict(record)
        try:
            with self.output_path.open("a",encoding="utf-8",newline="\n") as file:
                json.dump(record_dict,file,ensure_ascii=False,separators=(",",":"))
                file.write("\n")
        except OSError as exc:
            raise OSError(f"Could not append to JSONL file: "f"{self.output_path}\n{exc}") from exc
        return self.output_path

    # ============================================================
    # PRIMEVUL SERIALIZATION
    # ============================================================

    def _record_to_dict(self,record: PrimeVulRecord) -> dict[str, Any]:
        """
        Convert PrimeVulRecord to the exact PrimeVul-style
        JSONL schema.

        IMPORTANT:

        Only the seven official output fields are emitted.

        These are deliberately excluded:

            scenario_id
            scenario_context
            metadata
            vulnerabilities
            optimal_attack
            optimal_defense
            valid_attacks
            valid_defenses
            tags
            Red SFT information
            reasoning information

        `func` is copied EXACTLY as supplied by the builder.
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
    # RECORD VALIDATION
    # ============================================================

    def _validate_record(self,record: PrimeVulRecord,index: int) -> None:
        """
        Validate one PrimeVulRecord before serialization.
        """

        if not isinstance(record,PrimeVulRecord):
            raise TypeError(
                f"Record {index} is not a "
                f"PrimeVulRecord object."
            )

        if not isinstance(record.project,str):
            raise ValueError(
                f"Record {index}: "
                f"'project' must be a string."
            )

        if not record.project.strip():
            raise ValueError(
                f"Record {index}: "
                f"'project' cannot be empty."
            )

        if record.target not in (0,1):
            raise ValueError(
                f"Record {index}: "
                f"'target' must be 0 or 1, "
                f"got {record.target!r}."
            )

        # --------------------------------------------------------
        # FUNC
        # --------------------------------------------------------

        if not isinstance(record.func,str):
            raise ValueError(
                f"Record {index}: "
                f"'func' must be a string.")
        if not record.func.strip():
            raise ValueError(
                f"Record {index}: "
                f"'func' cannot be empty.")

        # --------------------------------------------------------
        # Commit ID
        # --------------------------------------------------------

        if (
            record.commit_id is not None
            and not isinstance(
                record.commit_id,
                str,
            )
        ):

            raise ValueError(
                f"Record {index}: "
                f"'commit_id' must be a string "
                f"or null."
            )

        # --------------------------------------------------------
        # CWE
        #
        # CWE may legitimately be:
        #
        #     null
        #     "CWE-79"
        #     ["CWE-79", "CWE-89"]
        #
        # Therefore do NOT force it to be a string.
        # --------------------------------------------------------

        self._validate_optional_cwe(
            record.cwe,
            index,
        )

        # --------------------------------------------------------
        # CVE
        # --------------------------------------------------------

        if (
            record.cve is not None
            and not isinstance(
                record.cve,
                str,
            )
        ):

            raise ValueError(
                f"Record {index}: "
                f"'cve' must be a string "
                f"or null."
            )

        # --------------------------------------------------------
        # CVE description
        # --------------------------------------------------------

        if (
            record.cve_desc is not None
            and not isinstance(
                record.cve_desc,
                str,
            )
        ):

            raise ValueError(
                f"Record {index}: "
                f"'cve_desc' must be a string "
                f"or null."
            )

        # --------------------------------------------------------
        # Leakage sanity check
        # --------------------------------------------------------

        self._validate_no_obvious_answer_leakage(
            record,
            index,
        )

    # ============================================================
    # CWE VALIDATION
    # ============================================================

    def _validate_optional_cwe(
        self,
        value: Any,
        index: int,
    ) -> None:
        """
        Validate PrimeVul CWE representations.

        Allowed:

            None
            string
            list/tuple of strings
        """

        if value is None:
            return

        if isinstance(
            value,
            str,
        ):
            return

        if isinstance(
            value,
            (list, tuple),
        ):

            for item in value:

                if not isinstance(
                    item,
                    str,
                ):

                    raise ValueError(
                        f"Record {index}: "
                        f"'cwe' list must contain "
                        f"only strings."
                    )

            return

        raise ValueError(
            f"Record {index}: "
            f"'cwe' must be null, a string, "
            f"or a list/tuple of strings."
        )

    # ============================================================
    # ANSWER-LEAKAGE SANITY CHECK
    # ============================================================

    def _validate_no_obvious_answer_leakage(
        self,
        record: PrimeVulRecord,
        index: int,
    ) -> None:
        """
        Catch obvious accidental metadata leakage into `func`.

        This is intentionally conservative.

        We do NOT attempt to detect legitimate source-code
        identifiers such as `target`, `attack`, etc.

        The check focuses on explicit dataset-answer markers.
        """

        func = record.func.lower()

        forbidden_markers = [
            "ground_truth:",
            "ground truth:",
            "expected_classification:",
            "expected classification:",
            "optimal_attack:",
            "optimal attack:",
            "optimal_defense:",
            "optimal defense:",
            "valid_attacks:",
            "valid attacks:",
            "valid_defenses:",
            "valid defenses:",
            "dataset_labels:",
            "red_sft:",
            "classification_label:",
            "classification label:",
        ]

        found = [
            marker
            for marker in forbidden_markers
            if marker in func
        ]

        if found:

            raise ValueError(
                f"Record {index}: "
                f"possible answer/metadata leakage "
                f"detected in 'func': {found}"
            )

    # ============================================================
    # OUTPUT PREPARATION
    # ============================================================

    def _prepare_output_path(
        self,
    ) -> None:
        """
        Create parent directory and enforce overwrite policy.
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
    # EXISTING JSONL VALIDATION
    # ============================================================

    def validate_existing_file(
        self,
        path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Validate an already-created JSONL file.

        Returns:

            total_records
            vulnerable
            safe
            errors
            valid
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

        errors: list[str] = []

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
                            f"Line "
                            f"{line_number}: "
                            f"{exc}"
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
    # RAW DICTIONARY VALIDATION
    # ============================================================

    def _validate_dict_record(
        self,
        record: dict[str, Any],
        line_number: int,
    ) -> None:
        """
        Validate one already-serialized JSON object.
        """

        if not isinstance(
            record,
            dict,
        ):

            raise ValueError(
                "record is not a JSON object"
            )

        # --------------------------------------------------------
        # Exact required schema
        # --------------------------------------------------------

        missing = [
            field
            for field in self.REQUIRED_FIELDS
            if field not in record
        ]

        if missing:

            raise ValueError(
                f"missing fields: {missing}"
            )

        # --------------------------------------------------------
        # PrimeVul fields only
        # --------------------------------------------------------

        unexpected = [
            field
            for field in record
            if field not in self.FIELD_ORDER
        ]

        if unexpected:

            raise ValueError(
                f"unexpected fields: "
                f"{unexpected}"
            )

        # --------------------------------------------------------
        # Target
        # --------------------------------------------------------

        if record["target"] not in (
            0,
            1,
        ):

            raise ValueError(
                f"invalid target: "
                f"{record['target']!r}"
            )

        # --------------------------------------------------------
        # Project
        # --------------------------------------------------------

        if not isinstance(
            record["project"],
            str,
        ):

            raise ValueError(
                "'project' must be a string"
            )

        if not record["project"].strip():

            raise ValueError(
                "'project' is empty"
            )

        # --------------------------------------------------------
        # FUNC
        # --------------------------------------------------------

        if not isinstance(
            record["func"],
            str,
        ):

            raise ValueError(
                "'func' must be a string"
            )

        if not record["func"].strip():

            raise ValueError(
                "'func' is empty"
            )

        # --------------------------------------------------------
        # Leakage check
        # --------------------------------------------------------

        fake_record = PrimeVulRecord(
            project=record["project"],
            commit_id=record["commit_id"],
            target=record["target"],
            func=record["func"],
            cwe=record["cwe"],
            cve=record["cve"],
            cve_desc=record["cve_desc"],
        )

        self._validate_no_obvious_answer_leakage(
            fake_record,
            line_number,
        )

    # ============================================================
    # STATISTICS
    # ============================================================

    def count_records(self,path: Path | None = None,) -> int:
        """
        Count non-empty JSONL records.
        """

        path = (Path(path) if path is not None else self.output_path)
        if not path.exists():
            return 0
        count = 0

        with path.open("r",encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    count += 1

        return count