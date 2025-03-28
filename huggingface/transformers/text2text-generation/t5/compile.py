import argparse
import os

from optimum.rbln import RBLNT5ForConditionalGeneration


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["t5-small", "t5-base", "t5-large", "t5-3b"],
        default="t5-base",
        help="(str) model type, Size of T5. [t5-small, t5-base, t5-large, t5-3b]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "google-t5/" + args.model_name

    # Compile and export
    model = RBLNT5ForConditionalGeneration.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
