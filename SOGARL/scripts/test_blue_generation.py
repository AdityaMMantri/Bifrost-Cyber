import torch

from configs import config
from src.generation.generator import Generator


def main():
    print("=" * 70)
    print("BLUE-ONLY GENERATION TEST")
    print("=" * 70)

    # We are explicitly testing independent loading.
    # Only ONE model will be loaded: Blue base + Blue LoRA.
    config.USE_SHARED_BACKBONE = False

    print(f"Base model : {config.BASE_MODEL_NAME}")
    print(f"Blue adapter: {config.BLUE_ADAPTER_PATH}")
    print("Loading BLUE only...")
    print()

    blue_generator = Generator.from_config(
        role="blue"
    )

    prompt = """
You are the Blue Team cybersecurity analyst.

Analyze the following situation and propose a defensive action.

A web application is suspected of being vulnerable to JWT forgery.
Explain what the defender should investigate and what defensive
measure should be applied.

Give a concise technical response.
"""

    results = blue_generator.generate(
        prompts=[prompt] * 3,
        temperature=0.7,
        generation_indices=[0, 1, 2],
        turn=2,
        generation_type="blue_test",
    )

    print()
    print("=" * 70)
    print("BLUE GENERATION RESULTS")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        print(f"\n--- BLUE RESPONSE {i} ---")
        print(result.response)

    print()
    print("=" * 70)
    print("BLUE-ONLY TEST COMPLETE")
    print("=" * 70)

    # Explicitly release the model before the process exits.
    del blue_generator
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()