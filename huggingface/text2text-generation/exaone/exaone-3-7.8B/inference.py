import os
from optimum.rbln import RBLNExaoneForCausalLM
from transformers import AutoTokenizer


def main():
    model_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"

    # Load compiled model
    model = RBLNExaoneForCausalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        trust_remote_code=True,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    text = "Explain who you are"
    messages = [
        {
            "role": "system",
            "content": "You are EXAONE model from LG AI Research, a helpful assistant.",
        },
        {"role": "user", "content": text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    # Generate tokens
    output_sequence = model.generate(input_ids, max_length=4096)

    input_len = input_ids.shape[-1]
    generated_texts = tokenizer.decode(
        output_sequence[0][input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Show text and result
    print("--- Text ---")
    print(text)
    print("--- Result ---")
    print(generated_texts)


if __name__ == "__main__":
    main()
