import os
import argparse
from optimum.rbln import RBLNWav2Vec2ForCTC


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["wav2vec2-base-960h"],
        default="wav2vec2-base-960h",
        help="(str) wav2vec model name, [wav2vec2-base-960h]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"facebook/{args.model_name}"

    # Compile and export
    model = RBLNWav2Vec2ForCTC.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=160005,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
