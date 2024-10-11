import os
import argparse
from optimum.rbln import RBLNWhisperForConditionalGeneration

from transformers import AutoProcessor, pipeline
from datasets import load_dataset
import torch


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["tiny", "small", "base", "medium", "large-v3"],
        default="base",
        help="(str) model type, Size of Whisper [tiny, small, base, medium, large-v3]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"openai/whisper-{args.model_name}"

    # Prepare inputs
    processor = AutoProcessor.from_pretrained(model_id)
    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample = dataset[0]["audio"]

    # Load compiled model
    model = RBLNWhisperForConditionalGeneration.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate with pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        return_timestamps="word",
        batch_size=1,
    )
    generate_kwargs = {"repetition_penalty": 1.3, "return_token_timestamps": True}

    with torch.no_grad():
        outputs = pipe(sample, generate_kwargs=generate_kwargs)

    print("--- Result ---")
    print("--Text--")
    print(outputs["text"])
    print("--Chunks--")
    print(outputs["chunks"])


if __name__ == "__main__":
    main()