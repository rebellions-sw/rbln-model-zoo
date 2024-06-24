import os
import argparse
from optimum.rbln import RBLNWhisperForConditionalGeneration

from transformers import WhisperProcessor
from datasets import load_dataset


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["tiny", "small", "base", "large"],
        default="base",
        help="(str) model type, Size of Whisper [tiny, small, base, large]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"openai/whisper-{args.model_name}"

    processor = WhisperProcessor.from_pretrained(model_id)

    # Prepare inputs
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
        trust_remote_code=True,
    )
    sample = ds[0]["audio"]
    input_features = processor(sample["array"], return_tensors="pt").input_features

    # Load compiled model
    model = RBLNWhisperForConditionalGeneration.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate transcription
    generated_ids = model.generate(input_features=input_features)
    transcription = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Result
    print("--- Result ---")
    print(transcription)


if __name__ == "__main__":
    main()
