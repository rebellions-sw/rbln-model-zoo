import os

from optimum.rbln import RBLNLlamaForCausalLM
from transformers import AutoTokenizer


def main():
    model_id = "meta-llama/Meta-Llama-3-8B"
    lora_id = "FinGPT/fingpt-mt_llama3-8b_lora"
    prompt = """Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}
Input: FINANCING OF ASPOCOMP 'S GROWTH Aspocomp is aggressively pursuing its growth strategy by increasingly focusing on technologically more demanding HDI printed circuit boards PCBs .
Answer: """  # noqa: E501

    # Load compiled model
    model_save_dir = f"{os.path.basename(model_id)}_{os.path.basename(lora_id)}"
    model = RBLNLlamaForCausalLM.from_pretrained(model_id=model_save_dir, export=False)

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate tokens
    output_sequence = model.generate(**inputs, max_length=512)
    input_len = inputs.input_ids.shape[-1]
    generated_texts = tokenizer.decode(output_sequence[0][input_len:], skip_special_tokens=True)

    # Show text and result
    print("--- text ---")
    print(prompt)
    print("--- Result ---")
    print(generated_texts)


if __name__ == "__main__":
    main()
