import argparse

import rebel  # RBLN Compiler
from transformers import TFBertForMaskedLM

import tensorflow as tf


def monkey_patch_transformers():
    import transformers.modeling_tf_pytorch_utils as tf_pytorch_utils

    original_func = tf_pytorch_utils.load_pytorch_state_dict_in_tf2_model

    def patched_func(*args, **kwargs):
        pt_state_dict = args[1]
        if hasattr(pt_state_dict, "keys") and hasattr(pt_state_dict, "get_tensor"):
            pt_state_dict = {
                key: pt_state_dict.get_tensor(key) for key in pt_state_dict.keys()
            }
            return original_func(args[0], pt_state_dict, *args[2:], **kwargs)
        else:
            return original_func(*args, **kwargs)

    tf_pytorch_utils.load_pytorch_state_dict_in_tf2_model = patched_func


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

    # Monkey patch transformers to support safetensors archive (transformers 4.57.1)
    monkey_patch_transformers()

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
