import argparse

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    # You need to set the parameters following rbln_config.json in the compiled model
    parser.add_argument(
        "-m",
        "--model_id",
        dest="model_id",
        action="store",
        help="Compiled model directory path",
        default="google/flan-t5-base",
    )
    parser.add_argument(
        "-l",
        "--max-sequence-length",
        dest="max_seq_len",
        type=int,
        action="store",
        help="Max sequence length",
        default=512,
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
    return args.model_id, args.max_seq_len, args.batch_size


def main():
    # Make sure the engine configuration
    # matches the parameters used during compilation.
    model_id, max_seq_len, batch_size = parse_args()
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
    )

    prompts = [
        "translate English to French: How are you?",
        "translate English to Spanish: I would like to order a coffee.",
    ]

    outputs = []
    for request_id, prompt in enumerate(prompts):
        encoder_prompt_token_ids = tokenizer.encode(prompt, truncation=True, max_length=200)
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
