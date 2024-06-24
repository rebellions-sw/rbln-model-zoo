import os
import torch
import argparse
from optimum.rbln import RBLNWav2Vec2ForCTC

from transformers import Wav2Vec2Processor
from datasets import load_dataset


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["wav2vec2-base-960h"],
        default="wav2vec2-base-960h",
        help="(str) wav2vec model name, [wav2vec2-base-960h]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"facebook/{args.model_name}"

    processor = Wav2Vec2Processor.from_pretrained(model_id)

    # Prepare inputs
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]
    input_values = processor(
        sample["array"],
        padding="max_length",
        max_length=160005,
        return_tensors="pt",
        truncation=True,
    ).input_values

    # Load compiled model
    model = RBLNWav2Vec2ForCTC.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate transcription
    output = model(input_values)
    predicted_ids = torch.argmax(output.logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # Result
    print("--- Result ---")
    print(transcription)


if __name__ == "__main__":
    main()
