from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Make sure the engine configuration
# matches the parameters used during compilation.
compiled_model_id = "google/flan-t5-base"
max_seq_len = 2048
batch_size = 4
num_input_prompt = 5

def generate_prompts(model_id: str):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild",
                           split="train").shuffle(seed=42)

    prompts = []
    for i in range(num_input_prompt):
        image = dataset[i]["image"]
        question = dataset[i]["question"]

        # Use simple QA template because BLIP2 don't have default chat template.
        text_prompt = (f"Question: {question}\n Answer:")

        prompts.append({
            "prompt": text_prompt,
            "multi_modal_data": {
                "image": [image]
            }
        })

    return prompts

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
  max_tokens=200,
)
inputs = generate_prompts(compiled_model_id)
outputs = []

for prompt in inputs:
    output = llm.generate(prompt, sampling_params=sampling_params)
    outputs.append(output)

for i, output in enumerate(outputs):
    output = output[0].outputs[0].text
    print(
        f"===================== Output {i} ==============================")
    print(output)
    print(
        "===============================================================\n"
    )