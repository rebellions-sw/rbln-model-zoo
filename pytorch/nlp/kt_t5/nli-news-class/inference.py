import argparse
import os
import pathlib
import sys
import rebel

import pandas as pd
import numpy as np

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(abs_path, "transformers_kt/src"))
from transformers import T5Tokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        required=True,
        help="(str) vocab path for decoding generated token",
    )
    parser.add_argument(
        "--rbln_model_path",
        type=str,
        default="./",
        help="(str) path of rbln compiled model",
    )

    return parser.parse_args()


def preprocess(input_text, max_source_length, tokenizer, pad_token_id=0):
    sample = tokenizer(input_text)["input_ids"]
    sources = (
        sample[:max_source_length]
        if len(sample) > max_source_length
        else sample + [pad_token_id] * (max_source_length - len(sample))
    )
    att_masks = [0 if y == pad_token_id else 1 for y in sources]

    return sources, att_masks


def main():
    args = parsing_argument()

    pad_token_id = 0
    input_seq_length = 128

    tokenizer = T5Tokenizer.from_pretrained(args.vocab_path)

    ## prepare dataset
    news_path = "df_daum_rebellions.csv"
    news_col = "article_text_clean"
    df_news = pd.read_csv(pathlib.Path(__file__).parent / news_path)
    all_sentences = df_news[news_col].to_list()
    input_sentences = all_sentences[:1]

    ## load module
    rbln_encoder_path = os.path.join(args.rbln_model_path, f"encoder_i_seq_{input_seq_length}.rbln")
    encoder_module = rebel.Runtime(rbln_encoder_path)

    for sample_num, sample in enumerate(input_sentences):
        input_sample = sample
        input_ids, input_attn_masks = preprocess(
            input_sample, input_seq_length, tokenizer, pad_token_id
        )

        enc_input_ids = np.ascontiguousarray(input_ids, dtype=np.long).reshape(1, -1)
        enc_attn_mask = np.ascontiguousarray(input_attn_masks, dtype=np.long).reshape(1, -1)
        rebel_final_output = encoder_module.run(enc_input_ids, enc_attn_mask)

        print_input = input_ids
        rebel_output = rebel_final_output[0]
        if tokenizer is not None:
            print_input = tokenizer.decode(print_input)

        print("-" * 10, "sample id : ", sample_num, "-" * 10)
        print("--- input sample ---")
        print(print_input)
        print("--- encoder output ---")
        print("=== REBEL ===")
        print(rebel_output)


if __name__ == "__main__":
    main()
