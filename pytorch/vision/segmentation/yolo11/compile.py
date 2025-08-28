import argparse
import os

import rebel
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"],
        required=True,
        help="model name",
    )

    args = parser.parse_args()

    return args


def compile_model(checkpoint):
    model = YOLO(checkpoint).model
    model.eval()

    input_info = [("input_np", [1, 3, 640, 640], torch.float32)]
    compiled_model = rebel.compile_from_torch(model, input_info)
    compiled_model.save(f"{os.path.splitext(os.path.basename(checkpoint))[0]}.rbln")


def main():
    args = parse_args()
    checkpoint = args.model_name + ".pt"
    compile_model(checkpoint)


if __name__ == "__main__":
    main()
