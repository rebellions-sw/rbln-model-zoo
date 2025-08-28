import argparse

from datasets import load_dataset

from vllm import LLM, SamplingParams


def generate_prompts(batch_size: int, model_id: str):
    dataset = load_dataset("distil-whisper/librispeech_asr-noise", "test-pub-noise", split="40")

    messages = [
        {
            "prompt": "<|startoftranscript|>",
            "multi_modal_data": {
                "audio": (dataset[i]["audio"]["array"], dataset[i]["audio"]["sampling_rate"])
            },
        }
        for i in range(batch_size)
    ]

    return messages


def parse_args():
    parser = argparse.ArgumentParser()
    # You need to set the parameters following rbln_config.json in the compiled model
    parser.add_argument(
        "-m",
        "--model_id",
        dest="model_id",
        action="store",
        help="Compiled model directory path",
        default="openai/whisper-tiny",
    )
    parser.add_argument(
        "-l",
        "--max-sequence-length",
        dest="max_seq_len",
        type=int,
        action="store",
        help="Max sequence length",
        default=131072,
    )
    parser.add_argument(
        "-k",
        "--kvcache-partition-len",
        dest="kvcache_partition_len",
        type=int,
        action="store",
        help="KV Cache length",
        default=16384,
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
    return (
        args.model_id,
        args.max_seq_len,
        args.kvcache_partition_len,
        args.batch_size,
        args.num_input_prompt,
    )


def main():
    # Make sure the engine configuration
    # matches the parameters used during compilation.
    model_id, max_seq_len, kvcache_partition_len, batch_size, num_input_prompt = parse_args()
    llm = LLM(
        model=model_id,
        device="auto",
        max_num_seqs=batch_size,
        max_num_batched_tokens=max_seq_len,
        max_model_len=max_seq_len,
        block_size=max_seq_len,
        limit_mm_per_prompt={"audio": 1},
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    inputs = generate_prompts(num_input_prompt, model_id)

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
