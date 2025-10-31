from transformers import AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
input_text = "Hey, are you conscious? Can you talk to me?"
batch_size = 1

# Prepare inputs
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
conversation = [[{"role": "user", "content": input_text}]] * batch_size
text = tokenizer.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=False
)
inputs = tokenizer(text, return_tensors="pt", padding=True)
input_ids = inputs.input_ids.numpy()

# Save the generated input binary files
input_ids.tofile("c_input_ids.bin")
