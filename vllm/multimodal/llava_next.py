import argparse

import requests
from PIL import Image
from transformers import LlavaNextProcessor

from vllm import LLM, SamplingParams

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
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
        default="llava-hf/llava-v1.6-mistral-7b-hf",
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
    args = parser.parse_args()
    return args.model_id, args.max_seq_len, args.kvcache_partition_len, args.batch_size


def main():
    model_id, max_seq_len, kvcache_partition_len, batch_size = parse_args()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    processor = LlavaNextProcessor.from_pretrained(model_id)

    llm = LLM(
        model=model_id,
        device="rbln",
        max_num_seqs=batch_size,
        max_num_batched_tokens=max_seq_len,
        max_model_len=max_seq_len,
        block_size=kvcache_partition_len,
        limit_mm_per_prompt={"image": 1},  # The maximum number to accept
    )

    url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = []
    for _ in range(10):
        data = {"prompt": prompt, "multi_modal_data": {"image": image}}
        inputs.append(data)

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    main()
