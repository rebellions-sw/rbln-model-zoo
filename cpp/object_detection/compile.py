import argparse

import rebel
import torch
from ultralytics import YOLO


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        choices=[
            "yolov8n",
            "yolov8s",
            "yolov8m",
            "yolov8l",
            "yolov8x",
        ],
        default="yolov8n",
        help="(str) Name of the YOLO model. [yolov8n, yolov8s, ...]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()

    IMAGE_SIZE = 640
    INPUT_SHAPE = [1, 3, IMAGE_SIZE, IMAGE_SIZE]

    yolo = YOLO(f"{args.model_name}.pt")
    model = yolo.model.eval()

    model(torch.zeros(INPUT_SHAPE))

    input_info = [("x", INPUT_SHAPE, "float32")]
    compiled_model = rebel.compile_from_torch(model, input_info)

    compiled_model.save(f"{args.model_name}.rbln")


if __name__ == "__main__":
    main()
