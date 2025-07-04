import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def generate_prompts(model_id: str, num_input_prompt: int):
    dataset = load_dataset("lmms-lab/llava-bench-in-the-wild", split="train").shuffle(seed=42)

    prompts = []
    for i in range(num_input_prompt):
        image = dataset[i]["image"]
        question = dataset[i]["question"]

        # Use simple QA template because BLIP2 don't have default chat template.
        text_prompt = f"Question: {question}\n Answer:"

        prompts.append({"prompt": text_prompt, "multi_modal_data": {"image": [image]}})

    return prompts


def parse_args():
    parser = argparse.ArgumentParser()
    # You need to set the parameters following rbln_config.json in the compiled model
    parser.add_argument(
        "-m",
        "--model_id",
        dest="model_id",
        action="store",
        help="Compiled model directory path",
        default="Salesforce/blip2-opt-2.7b",
    )
    parser.add_argument(
        "-l",
        "--max-sequence-length",
        dest="max_seq_len",
        type=int,
        action="store",
        help="Max sequence length",
        default=2048,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        type=int,
        action="store",
        help="Batch size",
        default=1,
    )
    parser.add_argument(
        "-n",
        "--num-input_prompt",
        dest="num_input_prompt",
        type=int,
        action="store",
        help="The number of prompts",
        default=1,
    )
    args = parser.parse_args()
    return args.model_id, args.max_seq_len, args.batch_size, args.num_input_prompt


def main():
    # Make sure the engine configuration
    # matches the parameters used during compilation.
    model_id, max_seq_len, batch_size, num_input_prompt = parse_args()
    llm = LLM(
        model=model_id,
        device="auto",
        max_num_seqs=batch_size,
        max_num_batched_tokens=max_seq_len,
        max_model_len=max_seq_len,
        block_size=max_seq_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sampling_params = SamplingParams(
        temperature=0.0,
        skip_special_tokens=True,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=200,
    )
    inputs = generate_prompts(model_id, num_input_prompt)
    outputs = []

    for prompt in inputs:
        output = llm.generate(prompt, sampling_params=sampling_params)
        outputs.append(output)

    for i, output in enumerate(outputs):
        output = output[0].outputs[0].text
        print(f"===================== Output {i} ==============================")
        print(output)
        print("===============================================================\n")


if __name__ == "__main__":
    main()
