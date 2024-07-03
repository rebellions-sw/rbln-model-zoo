import os
import torch
import argparse
from optimum.rbln import RBLNASTForAudioClassification

from transformers import AutoFeatureExtractor
from datasets import load_dataset


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["ast-finetuned-audioset-10-10-0.4593"],
        default="ast-finetuned-audioset-10-10-0.4593",
        help="(str) ast model name [ast-finetuned-audioset-10-10-0.4593]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"MIT/{args.model_name}"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    # Prepare inputs
    ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
    sample = ds[0]["audio"]
    input_values = feature_extractor(sample["array"], return_tensors="pt").input_values

    # Load compiled model
    model = RBLNASTForAudioClassification.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Classify audio
    output = model(input_values)
    class_ids = torch.argmax(output[0], dim=-1).item()
    label = model.config.id2label[class_ids]

    # Result
    print("--- Result ---")
    print(label)


if __name__ == "__main__":
    main()
