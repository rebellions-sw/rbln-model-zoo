import os
import argparse
from optimum.rbln import RBLNRobertaForSequenceClassification


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        choices=["emotion"],
        default="emotion",
        help="(str) model type,",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"cardiffnlp/twitter-roberta-base-{args.task}"

    # Compile and export
    model = RBLNRobertaForSequenceClassification.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=512,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
