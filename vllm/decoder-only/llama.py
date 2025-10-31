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
    parser.add_argument(
        "-m",
        "--model_id",
        dest="model_id",
        action="store",
        help="Compiled model directory path",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    args = parser.parse_args()
    return args.model_id


def stop_tokens(tokenizer):
    eot_id = next(
        (
            k
            for k, t in tokenizer.added_tokens_decoder.items()
            if t.content == "<|eot_id|>"
        ),
        None,
    )
    if eot_id is not None:
        return [tokenizer.eos_token_id, eot_id]
    else:
        return [tokenizer.eos_token_id]


def main():
    model_id = parse_args()
    llm = LLM(model=model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sampling_params = SamplingParams(
        temperature=0.0,
        skip_special_tokens=True,
        stop_token_ids=stop_tokens(tokenizer),
    )

    chat = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    outputs = llm.generate(chat, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
