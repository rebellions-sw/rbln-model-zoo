import argparse
import asyncio
import sys

from transformers import AutoTokenizer

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


# Please make sure the engine configurations match the parameters used when compiling.
def initialize(model_dir, max_seq_len, batch_size):
    engine_args = AsyncEngineArgs(
        model=model_dir,
        device="rbln",
        max_num_seqs=batch_size,
        max_num_batched_tokens=max_seq_len,
        max_model_len=max_seq_len,
        block_size=max_seq_len,
    )

    return engine_args


def stop_tokens(tokenizer):
    eot_id = next(
        (k for k, t in tokenizer.added_tokens_decoder.items() if t.content == "<|eot_id|>"), None
    )
    if eot_id is not None:
        return [tokenizer.eos_token_id, eot_id]
    else:
        return [tokenizer.eos_token_id]


async def run_single(engine, sampling_params, chat, request_id):
    results_generator = engine.generate(chat, sampling_params, request_id=request_id)
    final_result = None
    async for result in results_generator:
        # You can use the intermediate `result` here, if needed.
        final_result = result
    return final_result


async def run_multi(engine, sampling_params, chats):
    tasks = [
        asyncio.create_task(run_single(engine, sampling_params, chat, i))
        for (i, chat) in enumerate(chats)
    ]
    return [await task for task in tasks]


def parse_args():
    parser = argparse.ArgumentParser()

    # Examples from huggingface model_id : meta-llama/Meta-Llama-3-8B-Instruct
    parser.add_argument(
        "-m",
        "--model_dir",
        dest="model_dir",
        action="store",
        help='Model directory path(ex."Meta-Llama-3-8B-Instruct")',
    )
    parser.add_argument(
        "-l",
        "--max-sequence-length",
        dest="max_seq_len",
        type=int,
        action="store",
        help="Max sequence length(ex.8192)",
    )
    parser.add_argument(
        "-b", "--batch-size", dest="batch_size", type=int, action="store", help="Batch size(ex.1)"
    )
    if len(sys.argv) < 7:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args.model_dir, args.max_seq_len, args.batch_size


def main():
    model_dir, max_seq_len, batch_size = parse_args()
    engine_args = initialize(model_dir, max_seq_len, batch_size)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sampling_params = SamplingParams(
        temperature=0.0,
        skip_special_tokens=True,
        stop_token_ids=stop_tokens(tokenizer),
    )

    # Runs a single inference for an example
    conversation = [{"role": "user", "content": "What is the first letter of English alphabets?"}]
    chat = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    result = asyncio.run(run_single(engine, sampling_params, chat, "123"))
    print(result)

    # Runs multi inference for an example
    conversations = [
        [{"role": "user", "content": "What is the first letter of English alphabets?"}],
        [{"role": "user", "content": "What is the last letter of English alphabets?"}],
    ]
    chats = [
        tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        for conversation in conversations
    ]

    # Runs multiple inferences in parallel
    results = asyncio.run(run_multi(engine, sampling_params, chats))
    print(results)


if __name__ == "__main__":
    main()
