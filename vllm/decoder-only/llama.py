from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Make sure the engine configuration
# matches the parameters used during compilation.
compiled_model_id = "meta-llama/Meta-Llama-3-8B
max_seq_len = 8192
batch_size = 4

llm = LLM(
    model=compiled_model_id,
    device="rbln",
    max_num_seqs=batch_size,
    max_num_batched_tokens=max_seq_len,
    max_model_len=max_seq_len,
    block_size=max_seq_len,
)

tokenizer = AutoTokenizer.from_pretrained(compiled_model_id)

sampling_params = SamplingParams(
  temperature=0.0,
  skip_special_tokens=True,
  stop_token_ids=[tokenizer.eos_token_id],
)

conversation = [
    [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    [{"role": "user", "content": "What are the benefits of renewable energy?"}],
    [{"role": "user", "content": "Describe the process of photosynthesis."}],
    [{"role": "user", "content": "How does machine learning work?"}],
]

chat = tokenizer.apply_chat_template(
  conversation, 
  add_generation_prompt=True,
  tokenize=False
)

outputs = llm.generate(chat, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")