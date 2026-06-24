"""
metadata_loader.py

Loads metadata.json and converts it into the canonical
Metadata object used throughout the Dataset Builder.
"""

from pathlib import Path

from builders.models import Metadata
from utils.file_utils import FileUtils


class MetadataLoader:

    REQUIRED_RELEVANT_KEYS = ["relevant_files","relevant","important_files"]
    REQUIRED_NOISE_KEYS = ["noise_files","noise","distractor_files"]
    COMPETING_HYPOTHESIS_KEYS = ["competing_hypotheses","competing_hypothesis_files"]
    TECHNOLOGY_KEYS = ["technology_stack","technology","tech_stack"]
    TRUST_BOUNDARY_KEYS = ["trust_boundary","security_boundary","canonical_boundary"]


    def __init__(self,logger):
        self.logger = logger

    # Public API

    def load(self, metadata_path: Path) -> Metadata:

        self.logger.title("Loading metadata.json")
        self.logger.info(f"Reading : {metadata_path}")

        if not metadata_path.exists():

            raise FileNotFoundError(f"\nmetadata.json not found:\n{metadata_path}")

        data = FileUtils.read_json(metadata_path)

        relevant_files = self._extract_required_list(
            data,
            self.REQUIRED_RELEVANT_KEYS,
            "relevant_files"
        )

        noise_files = self._extract_required_list(
            data,
            self.REQUIRED_NOISE_KEYS,
            "noise_files"
        )

        competing_hypotheses = self._extract_optional(
            data,
            self.COMPETING_HYPOTHESIS_KEYS,
            []
        )

        technology_stack = self._extract_optional(
            data,
            self.TECHNOLOGY_KEYS,
            []
        )

        trust_boundary = self._extract_optional(
            data,
            self.TRUST_BOUNDARY_KEYS,
            ""
        )

        metadata = Metadata(

            scenario_name=metadata_path.parent.name,

            relevant_files=relevant_files,

            competing_hypotheses=competing_hypotheses,

            noise_files=noise_files,

            technology_stack=technology_stack,

            trust_boundary=trust_boundary,

            raw_data=data

        )

        self._print_summary(metadata)

        return metadata

    # =====================================================
    # Required Fields
    # =====================================================

    def _extract_required_list(self,data,aliases,field_name):

        for key in aliases:
            if key in data:

                value = data[key]

                if not isinstance(value, list):

                    raise TypeError(f"\nmetadata.json\n"f"Field '{key}' must be a list.")

                return value

        raise ValueError(f"\nRequired metadata field missing:\n"f"{field_name}")

    # =====================================================
    # Optional Fields
    # =====================================================

    def _extract_optional(self,data,aliases,default):

        for key in aliases:
            if key in data:
                return data[key]

        return default

    # =====================================================
    # Logging
    # =====================================================

    def _print_summary(self, metadata):

        self.logger.success("Metadata Loaded Successfully")

        self.logger.info(f"Scenario : {metadata.scenario_name}")

        self.logger.info(f"Relevant Files : {len(metadata.relevant_files)}")

        self.logger.info(f"Competing Hypotheses : {len(metadata.competing_hypotheses)}")

        self.logger.info(f"Noise Files : {len(metadata.noise_files)}")

        self.logger.info(f"Technology Entries : {len(metadata.technology_stack)}")

        if metadata.trust_boundary:

            self.logger.success(

                "Trust Boundary : Found"

            )

        else:

            self.logger.warning(

                "Trust Boundary : Not Found"

            )

# Reads metadata.json from a scenario.
# Example:
# Scenario021/metadata.json

# Checks that metadata.json exists.
# Stops the builder if it is missing.

# Reads the JSON data.
# Example:
# {
#   "relevant_files": ["security/Auth.py"],
#   "noise_files": ["README.md"]
# }

# Finds the required fields.
# Example:
# relevant_files -> ["security/Auth.py"]

# Also accepts alternate field names.
# Example:
# "relevant", "important_files" -> relevant_files

# Reads optional fields if present.
# Example:
# technology_stack, trust_boundary

# Creates a Metadata object.
# Example:
# Metadata(
#   scenario_name="Scenario021",
#   relevant_files=["security/Auth.py"],
#   noise_files=["README.md"]
# )

# Prints a summary.
# Example:
# Relevant Files : 4
# Noise Files : 6

# Returns the Metadata object to the PathResolver.