import argparse

import rebel
from torchvision import models


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        ],
        default="resnet18",
        help="(str) Name of the model. [resnet18, resnet34, ...]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()

    INPUT_SHAPE = [1, 3, 224, 224]

    weights = models.get_model_weights(args.model_name).DEFAULT
    model = getattr(models, args.model_name)(weights=weights).eval()

    input_info = [("x", INPUT_SHAPE, "float32")]
    compiled_model = rebel.compile_from_torch(model, input_info)

    compiled_model.save(f"{args.model_name}.rbln")


if __name__ == "__main__":
    main()
