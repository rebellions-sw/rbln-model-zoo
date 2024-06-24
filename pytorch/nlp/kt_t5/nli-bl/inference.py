import argparse
import os
import pathlib
import sys

import rebel

import numpy as np
import pickle as pkl


abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(abs_path, "transformers_kt/src"))
from transformers import T5Tokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="(str) vocab path for decoding generated token",
    )
    parser.add_argument(
        "--rbln_model_path",
        type=str,
        default="./",
        help="(str) path of rbln compiled model",
    )

    return parser.parse_args()


def preprocess(samples, max_source_length, pad_token_id=0, eos_token_id=3):
    samples = [104] + samples[0] + [105] + samples[1] + [eos_token_id]
    sources = [
        samples[: max_source_length - 1] + [eos_token_id]
        if len(samples) > max_source_length
        else samples + [pad_token_id] * (max_source_length - len(samples))  # for x in samples
    ]
    att_masks = [[0 if y == pad_token_id else 1 for y in x] for x in sources]

    return sources, att_masks


def main():
    args = parsing_argument()

    pad_token_id = 0
    eos_token_id = 3

    input_seq_length = 128
    vocab_path = args.vocab_path

    if vocab_path is not None:
        tokenizer = T5Tokenizer.from_pretrained(vocab_path)
    else:
        tokenizer = None

    ## prepare dataset
    with open(pathlib.Path(__file__).parent / "nli-bl_input_samples.pkl", "rb") as f:
        test_sample_dict = pkl.load(f)  # dict {sample_id : sample from dev.pkl}

    ## load module
    rbln_encoder_path = os.path.join(args.rbln_model_path, f"encoder_i_seq_{input_seq_length}.rbln")
    encoder_module = rebel.Runtime(rbln_encoder_path)

    for sample_iter, (sample_num, samples) in enumerate(test_sample_dict.items()):
        input_sample = samples[0]
        label = samples[1]
        input_ids, input_attn_masks = preprocess(
            input_sample, input_seq_length, pad_token_id, eos_token_id
        )

        enc_input_ids = np.ascontiguousarray(input_ids, dtype=np.long)
        enc_attn_mask = np.ascontiguousarray(input_attn_masks, dtype=np.long)

        rebel_final_output = encoder_module.run(enc_input_ids, enc_attn_mask)

        ########## end decoding #########
        print_input = input_ids[0]
        print_output = rebel_final_output[0]
        if tokenizer is not None:
            print_input = tokenizer.decode(print_input)
        print("-" * 10, "sample id : ", sample_num, "-" * 10)
        print("--- input sample ---")
        print(print_input)
        print("--- encoder output ---")
        print(print_output)
        print("--- label ---")
        print(label)


if __name__ == "__main__":
    main()
