import os
import argparse
from optimum.rbln import RBLNLlamaForCausalLM


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["Llama-2-13b-chat-hf"],
        default="Llama-2-13b-chat-hf",
        help="(str) model type, llama2-13b model name.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="(int) set tensor parallel size in llama model, default: 8",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"meta-llama/{args.model_name}"

    # Compile and export
    model = RBLNLlamaForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=4096,  # default "max_position_embeddings"
        rbln_tensor_parallel_size=args.tensor_parallel_size,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
