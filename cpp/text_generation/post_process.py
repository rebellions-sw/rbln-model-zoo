import numpy as np
import torch
from transformers import AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
input_text = "Hey, are you conscious? Can you talk to me?"
batch_size = 1

# Prepare inputs
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
conversation = [[{"role": "user", "content": input_text}]] * batch_size
text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt", padding=True)
input_ids = inputs.input_ids
input_len = inputs.input_ids.shape[-1]

output_sequence = torch.tensor(
    np.fromfile("c_text2text_generation_gen_id.bin", dtype=np.int64), dtype=torch.int64
)
generated_texts = tokenizer.decode(
    output_sequence[input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
)

print("--- input text ---")
print(input_text)
print("--- Decoded C Result ---")
print(generated_texts)
