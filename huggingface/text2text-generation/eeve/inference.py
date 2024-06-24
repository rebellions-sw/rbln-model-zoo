import os
import argparse

from optimum.rbln import RBLNLlamaForCausalLM
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["EEVE-Korean-Instruct-10.8B-v1.0"],
        default="EEVE-Korean-Instruct-10.8B-v1.0",
        help="(str) model type, eeve model name.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="한국의 수도는 어디인가요? 아래 선택지 중 골라주세요.\n\n(A) 경성\n(B) 부산\n(C) 평양\n(D) 서울\n(E) 전주",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"yanolja/{args.model_name}"

    # Load compiled model
    model = RBLNLlamaForCausalLM.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {prompt}\nAssistant:\n"
    inputs = tokenizer(
        prompt_template.format(prompt=args.text),
        return_tensors="pt",
        padding=True,
    )

    # Generate tokens
    output_sequence = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=4096,
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
