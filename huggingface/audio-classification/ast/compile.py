import os
import argparse
from optimum.rbln import RBLNASTForAudioClassification


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["ast-finetuned-audioset-10-10-0.4593"],
        default="ast-finetuned-audioset-10-10-0.4593",
        help="(str) ast model name [ast-finetuned-audioset-10-10-0.4593]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"MIT/{args.model_name}"

    # Compile and export
    model = RBLNASTForAudioClassification.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
