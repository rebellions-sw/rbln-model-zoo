import argparse
from transformers import TFBertForMaskedLM
import tensorflow as tf

import rebel  # RBLN Compiler


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

    # Set `max sequence length` of the compiled model
    MAX_SEQ_LEN = 128

    # Instantiate HuggingFace TensorFlow BERT-base model
    model = TFBertForMaskedLM.from_pretrained(model_name)
    func = tf.function(
        lambda input_ids, attention_mask, token_type_ids: model(
            input_ids, attention_mask, token_type_ids
        )
    )

    # Compile
    input_info = [
        ("input_ids", [1, MAX_SEQ_LEN], tf.int32),
        ("attention_mask", [1, MAX_SEQ_LEN], tf.int32),
        ("token_type_ids", [1, MAX_SEQ_LEN], tf.int32),
    ]
    compiled_model = rebel.compile_from_tf_function(func, input_info)

    # Save compiled results to disk
    compiled_model.save(f"bert-{args.model_name}.rbln")


if __name__ == "__main__":
    main()
