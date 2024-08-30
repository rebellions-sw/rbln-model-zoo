import os
import argparse
from optimum.rbln import RBLNGemmaForCausalLM


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["gemma-2b-it"],
        default="gemma-2b-it",
        help="(str) model type, gemma-2b model name.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"google/{args.model_name}"

    # Compile and export
    model = RBLNGemmaForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=8192,  # default "max_position_embeddings"
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
