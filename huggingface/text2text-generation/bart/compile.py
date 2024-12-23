import argparse
import os

from optimum.rbln import RBLNBartForConditionalGeneration


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["base", "large"],
        default="base",
        help="(str) model type, BART model size [base, large]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"facebook/bart-{args.model_name}"

    # Compile and export
    model = RBLNBartForConditionalGeneration.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
