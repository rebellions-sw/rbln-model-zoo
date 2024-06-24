import argparse
import os
import pathlib
import sys

import pickle as pkl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rbln_t5_utils import T5RBLNGeneration

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


def preprocess(samples, eos_token_id=3):
    if samples[-1] != eos_token_id:
        samples.append(eos_token_id)
    att_masks = [[1] * len(samples)]
    return [samples], att_masks


def main():
    args = parsing_argument()

    vocab_path = args.vocab_path

    if vocab_path is not None:
        tokenizer = T5Tokenizer.from_pretrained(vocab_path)
    else:
        tokenizer = None

    start_token_id = 0
    pad_token_id = 0
    eos_token_id = 3

    # define class
    model = T5RBLNGeneration(
        rbln_model_path=args.rbln_model_path,
        enc_seq_list=[384, 448, 512],
        max_dec_seq_length=256,
        batch_size=1,
        num_beams=0,
        start_token_id=start_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )

    with open(pathlib.Path(__file__).parent / "news_input_samples.pkl", "rb") as f:
        test_sample_dict = pkl.load(f)  # dict {sample_id : sample from dev.pkl}

    ## sample run
    for sample_iter, (sample_num, samples) in enumerate(test_sample_dict.items()):
        input_sample = samples[0]
        label = samples[1]

        # input shape : [1, inp_seq_len]
        # mask shape  : [1, inp_seq_len]
        input_ids, input_attn_masks = preprocess(input_sample, eos_token_id)

        # rebel_final_output : (batch, out_seq_len)
        rebel_final_output = model.generate(input_ids, input_attn_masks)

        ########## end decoding #########
        print_input = input_ids[0]
        print_output = rebel_final_output[0][1:]
        if tokenizer is not None:
            print_input = tokenizer.decode(print_input)
            print_output = tokenizer.decode(print_output)
            label = tokenizer.decode(label)
        print("-" * 10, "sample id : ", sample_num, "-" * 10)
        print("--- input sample ---")
        print(print_input)
        print("--- decoder output ---")
        print(print_output)
        print("--- label ---")
        print(label)


if __name__ == "__main__":
    main()
