import argparse
import torch
import rebel
import os
import sys

sys.path.append(os.path.join(sys.path[0], "yolov3"))
from yolov3.models.experimental import attempt_load


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov3",
        choices=["yolov3", "yolov3-tiny", "yolov3-spp"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name
    model = attempt_load(model_name + ".pt")
    model.eval()

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, 3, 640, 640], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
