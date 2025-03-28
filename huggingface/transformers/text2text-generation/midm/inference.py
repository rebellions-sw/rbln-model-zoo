import argparse
import os

from optimum.rbln import RBLNMidmLMHeadModel
from transformers import AutoConfig, AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        default="###User;AI란?\n###Midm;",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "KT-AI/midm-bitext-S-7B-inst-v1"

    # Load compiled model
    config = AutoConfig.from_pretrained(os.path.basename(model_id), trust_remote_code=True)
    model = RBLNMidmLMHeadModel.from_pretrained(
        model_id=os.path.basename(model_id),
        config=config,
        trust_remote_code=True,
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    inputs = tokenizer(args.text, return_tensors="pt", padding=True)

    # Generate tokens
    output_sequence = model.generate(
        input_ids=inputs.input_ids[..., :-1],
        attention_mask=inputs.attention_mask[..., :-1],
        max_length=8192,
    )

    input_len = inputs.input_ids[..., :-1].shape[-1]
    generated_texts = tokenizer.decode(
        output_sequence[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Show text and result
    print("--- text ---")
    print(args.text)
    print("--- Result ---")
    print(generated_texts)


if __name__ == "__main__":
    main()
