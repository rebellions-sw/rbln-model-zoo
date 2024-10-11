import os
import argparse
from optimum.rbln import RBLNMistralForCausalLM


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["Mistral-7B-Instruct-v0.3"],
        default="Mistral-7B-Instruct-v0.3",
        help="(str) model type, mistral model name.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="(int) set tensor parallel size in mistral model, default: 8",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"mistralai/{args.model_name}"

    # Compile and export
    model = RBLNMistralForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=32768,  # default "max_position_embeddings"
        rbln_tensor_parallel_size=args.tensor_parallel_size,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
