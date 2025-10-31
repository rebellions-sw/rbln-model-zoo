import argparse

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, SamplingParams

# Huggingface model : Qwen/Qwen2.5-VL-7B-Instruct
# If the video is too long
# set `VLLM_ENGINE_ITERATION_TIMEOUT_S` to a higher timeout value.
VIDEO_URLS = [
    "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4",
]


def generate_prompts_video(model_id: str):
    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    video_nums = len(VIDEO_URLS)
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": VIDEO_URLS[i],
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            },
        ]
        for i in range(video_nums)
    ]

    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    arr_video_inputs = []
    arr_video_kwargs = []
    for i in range(video_nums):
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages[i], return_video_kwargs=True
        )
        arr_video_inputs.append(video_inputs)
        arr_video_kwargs.append(video_kwargs)

    return [
        {
            "prompt": text,
            "multi_modal_data": {
                "video": video_inputs,
            },
            "mm_processor_kwargs": {
                "min_pixels": 1024 * 14 * 14,
                "max_pixels": 5120 * 14 * 14,
                **video_kwargs,
            },
        }
        for text, video_inputs, video_kwargs in zip(
            texts, arr_video_inputs, arr_video_kwargs
        )
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
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    args = parser.parse_args()
    return args.model_id


def main():
    # Make sure the engine configuration
    # matches the parameters used during compilation.
    model_id = parse_args()
    llm = LLM(model=model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sampling_params = SamplingParams(
        temperature=0,
        ignore_eos=False,
        skip_special_tokens=True,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=200,
    )

    inputs = generate_prompts_video(model_id)
    outputs = llm.generate(inputs, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
