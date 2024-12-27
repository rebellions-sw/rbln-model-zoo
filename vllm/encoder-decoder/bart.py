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


async def run_multi(engine, sampling_params, prompts):
    tasks = [
        asyncio.create_task(run_single(engine, sampling_params, prompt, i))
        for (i, prompt) in enumerate(prompts)
    ]
    return [await task for task in tasks]


def parse_args():
    parser = argparse.ArgumentParser()

    # Examples from huggingface model_id : lucadiliello/bart-small
    parser.add_argument(
        "-m",
        "--model_dir",
        dest="model_dir",
        action="store",
        help='Model directory path(ex."bart-small")',
    )
    parser.add_argument(
        "-l",
        "--max-sequence-length",
        dest="max_seq_len",
        type=int,
        action="store",
        help="Max sequence length(ex.512)",
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
        ignore_eos=False,
        skip_special_tokens=True,
        stop_token_ids=stop_tokens(tokenizer),
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Runs a single inference for an example
    INPUT_PROMPT = "UN Chief Says There Is No <mask> in Syria"
    result = asyncio.run(run_single(engine, sampling_params, INPUT_PROMPT, "123"))
    print(result)

    # Runs multi inference for an example
    INPUT_PROMPTS = [
        "The capital city of France is <mask>.",
        "She opened the door to find a <mask> waiting on the other side.",
        "The <mask> was so <mask> that everyone started clapping.",
        "There are <mask> continents on Earth.",
        "<mask> is the largest planet in our solar system.",
        "To bake a cake, you need flour, sugar, eggs, and <mask>.",
        "He <mask> the ball over the fence and scored a home run.",
        "After hearing the news, she felt <mask> and started crying.",
        "<mask> is known for his theory of relativity.",
        "The Great Wall of China is located in <mask>.",
    ]
    results = asyncio.run(run_multi(engine, sampling_params, INPUT_PROMPTS))
    print(results)


if __name__ == "__main__":
    main()
