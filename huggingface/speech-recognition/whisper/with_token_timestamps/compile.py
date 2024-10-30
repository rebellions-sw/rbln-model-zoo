import os
import argparse
from optimum.rbln import RBLNWhisperForConditionalGeneration


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["tiny", "small", "base", "medium", "large-v3", "large-v3-turbo"],
        default="base",
        help="(str) model type, [tiny, small, base, medium, large-v3, large-v3-turbo]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"openai/whisper-{args.model_name}"

    # Compile and export
    model = RBLNWhisperForConditionalGeneration.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_token_timestamps=True,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
