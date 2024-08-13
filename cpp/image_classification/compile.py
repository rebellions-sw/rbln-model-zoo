from argparse import ArgumentParser

from torchvision import models
import rebel
import torch

parser = ArgumentParser()
parser.add_argument("-m", "--model-name", choices=[
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
], help="Model name")
args = parser.parse_args()

weights = models.get_model_weights(args.model_name).DEFAULT
model = getattr(models, args.model_name)(weights=weights).eval()

compiled_model = rebel.compile_from_torch(model, [("x", [1, 3, 224, 224], "float32")])
compiled_model.save(f"{args.model_name}.rbln")
