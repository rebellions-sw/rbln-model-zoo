from argparse import ArgumentParser

from ultralytics import YOLO
import rebel
import torch

parser = ArgumentParser()
parser.add_argument("-m", "--model-name", choices=[
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
], help="YOLO model name")
args = parser.parse_args()

yolo = YOLO(f"{args.model_name}.pt")
model = yolo.model.eval()
model(torch.zeros(1, 3, 640, 640))
compiled_model = rebel.compile_from_torch(model, [("x", [1, 3, 640, 640], "float32")])
compiled_model.save(f"{args.model_name}.rbln")
