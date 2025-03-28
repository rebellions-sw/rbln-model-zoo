import argparse
import os

from optimum.rbln import RBLNGemmaForCausalLM
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="Write a hello world program",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "google/gemma-2b-it"

    # Load compiled model
    model = RBLNGemmaForCausalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    conversation = [{"role": "user", "content": args.text}]
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Generate tokens
    output_sequence = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=8192,
    )

    input_len = inputs.input_ids.shape[-1]
    generated_texts = tokenizer.decode(
        output_sequence[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Show text and result
    print("--- Text ---")
    print(args.text)
    print("--- Result ---")
    print(generated_texts)


if __name__ == "__main__":
    main()
