import os
import argparse
from optimum.rbln import RBLNMidmLMHeadModel
from transformers import AutoConfig


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="midm-bitext-S-7B-inst-v1",
        help="(str) model type, llama3-8b model name.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"KT-AI/{args.model_name}"

    # Compile and export
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = RBLNMidmLMHeadModel.from_pretrained(
        model_id=model_id,
        config=config,
        export=True,  # export a PyTorch model to RBLN model with optimum
        trust_remote_code=True,
        rbln_batch_size=1,
        rbln_max_seq_len=8192,  # default "max_position_embeddings"
        rbln_tensor_parallel_size=4,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
