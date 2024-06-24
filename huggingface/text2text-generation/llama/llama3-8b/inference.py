import os
import argparse

from optimum.rbln import RBLNLlamaForCausalLM
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["Meta-Llama-3-8B-Instruct"],
        default="Meta-Llama-3-8B-Instruct",
        help="(str) model type, llama3-8b model name.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hey, are you conscious? Can you talk to me?",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"meta-llama/{args.model_name}"

    # Load compiled model
    model = RBLNLlamaForCausalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
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
    print("--- text ---")
    print(args.text)
    print("--- Result ---")
    print(generated_texts)


if __name__ == "__main__":
    main()
