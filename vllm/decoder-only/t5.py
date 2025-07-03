from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Make sure the engine configuration
# matches the parameters used during compilation.
compiled_model_id = "google/flan-t5-base"
max_seq_len = 512
batch_size = 4

llm = LLM(
    model=compiled_model_id,
    device="auto",
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

prompts = [
    "translate English to French: How are you?",
    "translate English to Spanish: I would like to order a coffee."
]

outputs = []
for request_id, prompt in enumerate(prompts):
    encoder_prompt_token_ids = tokenizer.encode(
        prompt, truncation=True, max_length=200)
    input_prompt={
        "encoder_prompt": {
            "prompt_token_ids": encoder_prompt_token_ids,
        },
        "decoder_prompt": ""
    }
    output = llm.generate(input_prompt, sampling_params)
    outputs.append(output)

for output in outputs:
    prompt = output[0].prompt
    generated_text = output[0].outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")