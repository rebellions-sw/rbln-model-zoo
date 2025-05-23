import argparse

import rebel  # RBLN Compiler
from transformers import TFBertForQuestionAnswering

import tensorflow as tf


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
    model_name = (
        "HomayounSadri/bert-base-uncased-finetuned-squad-v2"
        if args.model_name == "base"
        else "bert-large-uncased-whole-word-masking-finetuned-squad"
    )
    # Set `max sequence length` of the compiled model
    MAX_SEQ_LEN = 384

    # Instantiate HuggingFace TensorFlow BERT-base model
    model = TFBertForQuestionAnswering.from_pretrained(model_name)

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
    compiled_model = rebel.compile_from_tf_function(
        func, input_info, outputs=["Identity_1", "Identity"]
    )

    # Save compiled results to disk
    compiled_model.save(f"bert-{args.model_name}.rbln")


if __name__ == "__main__":
    main()
