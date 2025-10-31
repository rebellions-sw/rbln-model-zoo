import argparse

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_id",
        dest="model_id",
        action="store",
        help="Compiled model directory path",
        default="google/flan-t5-base",
    )
    args = parser.parse_args()
    return args.model_id


def main():
    # V1 does not support encoder-decoder models anymore.
    # Make sure use `VLLM_USE_V1=0` in the environment variables.
    model_id = parse_args()
    llm = LLM(model=model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sampling_params = SamplingParams(
        temperature=0.0,
        skip_special_tokens=True,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    prompts = [
        "translate English to French: How are you?",
        "translate English to Spanish: I would like to order a coffee.",
    ]

    outputs = []
    for request_id, prompt in enumerate(prompts):
        encoder_prompt_token_ids = tokenizer.encode(
            prompt, truncation=True, max_length=200
        )
        input_prompt = {
            "encoder_prompt": {
                "prompt_token_ids": encoder_prompt_token_ids,
            },
            "decoder_prompt": "",
        }
        output = llm.generate(input_prompt, sampling_params)
        outputs.append(output)

    for output in outputs:
        prompt = output[0].prompt
        generated_text = output[0].outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
