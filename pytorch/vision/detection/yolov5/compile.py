import argparse

import rebel
import torch


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov5s",
        choices=["yolov5s", "yolov5n", "yolov5m", "yolov5l", "yolov5x"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    model = torch.hub.load("ultralytics/yolov5", model_name)
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
