import argparse
from transformers import BertTokenizer
import tensorflow as tf

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
    text = "The capital of Korea is [MASK]."
    inputs = tokenizer(text, return_tensors="tf", padding="max_length", max_length=MAX_SEQ_LEN)

    input_ids = inputs["input_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()
    token_type_ids = inputs["token_type_ids"].numpy()

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"bert-{args.model_name}.rbln")

    # Run inference
    out = module.run(input_ids, attention_mask, token_type_ids)

    # Decoding final logit to text
    mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
    selected_logits = tf.gather_nd(out[0], indices=mask_token_index)
    predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
    print("Masked word is [", tokenizer.decode(predicted_token_id), "].")


if __name__ == "__main__":
    main()
