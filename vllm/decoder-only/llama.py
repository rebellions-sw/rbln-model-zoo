import argparse

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

conversation = [
    [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    [{"role": "user", "content": "What are the benefits of renewable energy?"}],
    [{"role": "user", "content": "Describe the process of photosynthesis."}],
    [{"role": "user", "content": "How does machine learning work?"}],
]


def parse_args():
    parser = argparse.ArgumentParser()
    # You need to set the parameters following rbln_config.json in the compiled model
    parser.add_argument(
        "-m",
        "--model_id",
        dest="model_id",
        action="store",
        help="Compiled model directory path",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "-l",
        "--max-sequence-length",
        dest="max_seq_len",
        type=int,
        action="store",
        help="Max sequence length",
        default=32768,
    )
    parser.add_argument(
        "-k",
        "--kvcache-partition-len",
        dest="kvcache_partition_len",
        type=int,
        action="store",
        help="KV Cache length",
        default=32768,
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
    args = parser.parse_args()
    return args.model_id, args.max_seq_len, args.kvcache_partition_len, args.batch_size


def stop_tokens(tokenizer):
    eot_id = next(
        (k for k, t in tokenizer.added_tokens_decoder.items() if t.content == "<|eot_id|>"), None
    )
    if eot_id is not None:
        return [tokenizer.eos_token_id, eot_id]
    else:
        return [tokenizer.eos_token_id]


def main():
    # Make sure the engine configuration
    # matches the parameters used during compilation.
    model_id, max_seq_len, kvcache_partition_len, batch_size = parse_args()
    llm = LLM(
        model=model_id,
        device="auto",
        max_num_seqs=batch_size,
        max_num_batched_tokens=max_seq_len,
        max_model_len=max_seq_len,
        block_size=kvcache_partition_len,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sampling_params = SamplingParams(
        temperature=0.0,
        skip_special_tokens=True,
        stop_token_ids=stop_tokens(tokenizer),
    )

    chat = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    outputs = llm.generate(chat, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
