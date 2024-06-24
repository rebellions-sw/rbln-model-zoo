import argparse
import os
import torch
from pathlib import Path

import rebel  # RBLN Compiler

from encoder_only_models import KTULMEncoderForSequenceClassificationSimple


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weight_path", type=str, help="(str) path of weight file", required=True)
    parser.add_argument(
        "--rbln_model_path",
        type=str,
        default="./",
        help="(str) base directory to save compiled model",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()

    model_path = Path(args.rbln_model_path)
    if not os.path.isdir(model_path):
        model_path.mkdir(parents=True)

    model = KTULMEncoderForSequenceClassificationSimple.from_pretrained(
        args.weight_path, torchscript=True
    )
    input_seq_length = 128

    # Compile
    input_info = [
        ("input_ids", [1, input_seq_length], torch.long),
        ("attention_mask", [1, input_seq_length], torch.long),
    ]

    compiled_model = rebel.compile_from_torch(model, input_info)

    encoder_save_path = os.path.join(args.rbln_model_path, f"encoder_i_seq_{input_seq_length}.rbln")

    # Save compiled results to disk
    compiled_model.save(encoder_save_path)


if __name__ == "__main__":
    main()
