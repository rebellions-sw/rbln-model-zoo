import argparse
from transformers import BertTokenizer, pipeline
import torch

import rebel  # RBLN Runtime


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["base", "large"],
        default="base",
        help="(str) type, Size of BERT. [base or large]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = "bert-" + args.model_name + "-uncased"
    MAX_SEQ_LEN = 128

    # Prepare input text sequence for masked language modeling
    tokenizer = BertTokenizer.from_pretrained(model_name)
    text = "the color of rose is [MASK]."
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=MAX_SEQ_LEN)

    # Prepare input tensors as numpy array
    input_ids = inputs["input_ids"].numpy()
    token_type_ids = inputs["token_type_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"bert-{args.model_name}.rbln")

    # Run inference
    out = module.run(input_ids, attention_mask, token_type_ids)

    # Decoding final logit to text
    unmasker = pipeline("fill-mask", model=model_name)
    print(unmasker.postprocess({"input_ids": inputs["input_ids"], "logits": torch.from_numpy(out)}))


if __name__ == "__main__":
    main()
