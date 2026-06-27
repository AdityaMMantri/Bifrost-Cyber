"""
build_blue_jsonl.py

Main entry point for building the Blue Team JSONL dataset.
"""

from pathlib import Path

from config import *

from builders.logger import Logger
from builders.metadata_loader import MetadataLoader
from builders.scenario_loader import ScenarioLoader
from builders.blue_sft_loader import BlueSFTLoader
from builders.path_resolver import PathResolver
from builders.code_loader import CodeLoader
from builders.prompt_builder import PromptBuilder
from builders.jsonl_writer import JSONLWriter

from utils.file_utils import FileUtils
from utils.random_utils import RandomUtils


def get_scenarios():

    if BUILD_ALL_SCENARIOS:

        return FileUtils.get_scenario_folders(DATASET_ROOT)

    folders = []

    for name in SCENARIOS_TO_BUILD:

        folders.append(

            DATASET_ROOT / name

        )

    return folders


def main():

    logger = Logger(LOG_DIR / "build.log")

    random_utils = RandomUtils(RANDOM_SEED)

    metadata_loader = MetadataLoader(logger)

    scenario_loader = ScenarioLoader(logger)

    blue_loader = BlueSFTLoader(logger)

    resolver = PathResolver(logger)

    code_loader = CodeLoader(logger)

    prompt_builder = PromptBuilder(

        SYSTEM_PROMPT_FILE,

        PROMPT_TEMPLATE_DIR,

        logger

    )

    writer = JSONLWriter()

    scenarios = get_scenarios()

    logger.title("Dataset Builder")

    logger.info(f"Found {len(scenarios)} scenarios")

    for scenario_folder in scenarios:

        logger.title(f"Processing {scenario_folder.name}")

        # ------------------------------------
        # Paths
        # ------------------------------------

        metadata_path = scenario_folder / "metadata.json"

        scenario_path = scenario_folder / "scenario.md"

        blue_path = scenario_folder / "blue_sft.json"

        # ------------------------------------
        # Load metadata
        # ------------------------------------

        metadata = metadata_loader.load(

            metadata_path

        )

        # ------------------------------------

        scenario = scenario_loader.load(

            scenario_path

        )

        # ------------------------------------

        blue_examples = blue_loader.load(

            blue_path

        )

        # ------------------------------------
        # Select Noise Files
        # ------------------------------------

        selected_noise = random_utils.sample(

            metadata.noise_files,

            random_utils.random.randint(

                MIN_NOISE_FILES,

                MAX_NOISE_FILES

            )

        )

        # ------------------------------------
        # Resolve Paths
        # ------------------------------------

        resolved = resolver.resolve_files(

            scenario_folder,

            metadata.relevant_files,

            selected_noise

        )

        # ------------------------------------
        # Load Code
        # ------------------------------------

        code_files = code_loader.load(

            resolved

        )

        # ------------------------------------
        # Build every SFT example
        # ------------------------------------

        for example in blue_examples:

            logger.title(

                f"{scenario_folder.name} - {example.example_id}"

            )

            prompt = prompt_builder.build(

                scenario,

                code_files,

                example.question

            )

            writer.write_example(

                TRAIN_JSONL,

                scenario_folder.name,

                example.example_id,

                prompt,

                example.answer

            )

    logger.title("Finished")

    logger.success(

        "Dataset successfully generated."

    )


if __name__ == "__main__":

    main()