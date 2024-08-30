import os
import argparse

from optimum.rbln import RBLNMistralForCausalLM
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["Mistral-7B-Instruct-v0.3"],
        default="Mistral-7B-Instruct-v0.3",
        help="(str) model type, mistral model name.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Who are you?",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"mistralai/{args.model_name}"

    # Load compiled model
    model = RBLNMistralForCausalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    conversation = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {"role": "user", "content": args.text},
    ]
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Generate tokens
    output_sequence = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=32768,
        pad_token_id=tokenizer.eos_token_id,
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
