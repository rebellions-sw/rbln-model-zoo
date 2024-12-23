import argparse
import urllib.request

import cv2
import numpy as np
import rebel
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression as nms, scale_boxes, scale_coords
from ultralytics.utils.plotting import Annotator


def preprocess(image):
    preprocess_input = LetterBox(new_shape=(640, 640))(image=image)
    preprocess_input = preprocess_input.transpose((2, 0, 1))[::-1]
    preprocess_input = np.ascontiguousarray(preprocess_input, dtype=np.float32)
    preprocess_input = preprocess_input[None]
    preprocess_input /= 255

    return preprocess_input


def postprocess(outputs, input_image, origin_image):
    pred = nms(torch.from_numpy(outputs), 0.25, 0.7, False, max_det=300, nc=1)[0]
    pred[:, :4] = scale_boxes(input_image.shape[2:], pred[:, :4], origin_image.shape)
    kpts_shape = [17, 3]
    pred_kpts = pred[:, 6:].view(len(pred), *kpts_shape) if len(pred) else pred[:, 6:]
    pred_kpts = scale_coords(input_image.shape[2:], pred_kpts, origin_image.shape)

    annotator = Annotator(origin_image, line_width=3)
    for *xyxy, conf, _ in reversed(pred[:, :6]):
        label = f"people {conf:.2f}"
        annotator.box_label(xyxy, label=label)
    for k in reversed(pred_kpts):
        annotator.kpts(k, shape=origin_image.shape[:2], radius=5, kpt_line=True)

    return annotator.result()


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov8s-pose",
        choices=["yolov8s-pose", "yolov8n-pose", "yolov8m-pose", "yolov8l-pose", "yolov8x-pose"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    # Prepare input image
    img_url = "https://ultralytics.com/images/bus.jpg"
    img_path = "./bus.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())
    img = cv2.imread(img_path)
    batch = preprocess(img)

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"{model_name}.rbln")

    rebel_result = module.run(batch)

    rebel_post_output = postprocess(rebel_result[0], batch, img)
    cv2.imwrite(f"bus_{model_name}.jpg", rebel_post_output)


if __name__ == "__main__":
    main()
