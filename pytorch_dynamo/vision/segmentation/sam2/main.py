import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import sys
import os
import cv2

import subprocess

if torch.__version__ >= "2.5.0":
    torch._dynamo.config.inline_inbuilt_nn_modules = False
    
sys.path.insert(0, os.path.join(sys.path[0], "sam2"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend

CFG_CONFIG = {
    "sam2.1_hiera_large": {
        "config_file": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "sam2.1_hiera_large.pt",
    },
}


def get_args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference script for sam2 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sam2.1_hiera_large",
        help="Model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        choices=[1, 2],
        default=1,
        help="Batch size.",
    )
    args, _ = parser.parse_known_args()
    return args


def run_download_script():
    # Download the model checkpoint script
    print("Downloading model checkpoint script...")
    try:
        subprocess.run(
            [
                "curl",
                "-O",
                "https://raw.githubusercontent.com/facebookresearch/sam2/main/checkpoints/download_ckpts.sh",
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        # Run the downloaded script
        result = subprocess.run(
            ["bash", "download_ckpts.sh"],
            check=True,
            text=True,
            capture_output=True,
        )
        print("Script output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred while running the checkpoint script:", e.stderr)


def download_file(url, output_path):
    print(f"Downloading file from {url} to {output_path}...")
    try:
        result = subprocess.run(
            ["curl", "-o", output_path, url],
            check=True,
            text=True,
            capture_output=True,
        )
        print("File downloaded successfully.", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e.stderr}")


class WrapperSAM2ImagePredictor(SAM2ImagePredictor):
    def set_image(self, image: Union[np.ndarray, Image.Image]) -> None:
        batch_size = image.shape[0]

        backbone_out = self.model.forward_image(image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        vision_feats[-1] += self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        return feats[-1], feats[:-1]

    def set_image_wrapper(self, image_list, set_image):
        self.reset_predictor()
        self._orig_hw = []
        for image in image_list:
            self._orig_hw.append(image.shape[:2])

        img_batch = [self._transforms(img) for img in image_list]
        img_batch = torch.stack(img_batch, dim=0)

        feats = set_image(img_batch)

        self._features = {"image_embed": feats[0], "high_res_feats": feats[1]}
        self._is_image_set = True

        if len(image_list) == 1:
            self._is_batch = False
        else:
            self._is_batch = True


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    ax.axis("off")


# Initialize phase
def initialize(predictor, batch_size):
    set_image = torch.compile(
        predictor.set_image,
        backend="rbln",
        options={"cache_dir": "./.cache"},
        dynamic=False,
    )

    # warmup the predictor to pre-load the cached model
    set_image(torch.ones(batch_size, 3, 1024, 1024))

    return set_image, predictor


# Run phase
def run(set_image, predictor, image_list, pts_batch, labels_batch):
    predictor.set_image_wrapper(image_list, set_image)

    if len(image_list) == 1:
        masks_batch, scores_batch, logits_batch = predictor.predict(pts_batch[0], labels_batch[0])
    else:
        masks_batch, scores_batch, logits_batch = predictor.predict_batch(
            pts_batch, labels_batch, box_batch=None, multimask_output=True
        )

    return masks_batch, scores_batch, logits_batch


def save_segmented_image(
    image_batch,
    pts_batch,
    masks_batch,
    labels_batch,
    scores_batch,
):
    if len(image_batch) == 1:
        masks_batch = [masks_batch]
        scores_batch = [scores_batch]

    # Select the best single mask per object
    best_masks = []
    for masks, scores in zip(masks_batch, scores_batch):
        best_masks.append(masks[range(len(masks)), np.argmax(scores, axis=-1)])

    for i, (image, points, best_mask, labels, score) in enumerate(
        zip(image_batch, pts_batch, best_masks, labels_batch, scores_batch)
    ):
        plt.figure(figsize=(10, 10))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.imshow(gray_image, cmap="gray")

        for mask in best_mask:
            show_mask(mask, plt.gca(), random_color=True)

        plt.savefig(
            f"segmented_object_{i}.jpg",
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close()


def main():
    # Set up arguments
    opts = get_args_parser()
    cfg = CFG_CONFIG[opts.model_name]

    if not os.path.isfile(cfg["checkpoint"]):
        print("Checkpoint not found. Downloading model checkpoint script...")
        run_download_script()

    images_dir = "images"
    if not os.path.isdir(images_dir):
        print("Input image not found. Downloading input image...")

        os.makedirs(images_dir, exist_ok=True)
        print(f"Directory '{images_dir}' created or already exists.")

        download_file(
            "https://raw.githubusercontent.com/facebookresearch/sam2/main/notebooks/images/groceries.jpg",
            os.path.join(images_dir, "groceries.jpg"),
        )
        download_file(
            "https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/notebooks/images/truck.jpg",
            os.path.join(images_dir, "trunk.jpg"),
        )

    model = build_sam2(cfg["config_file"], cfg["checkpoint"], device="cpu")
    predictor = WrapperSAM2ImagePredictor(model)
    set_image, predictor = initialize(predictor, opts.batch_size)

    # Load images and points
    image_batch = [np.array(Image.open("images/groceries.jpg").convert("RGB"))]
    pts_batch = [np.array([[[400, 300]], [[630, 300]]])]
    labels_batch = [np.array([[1], [1]])]

    # Load second image and points based on opts.batch_size
    for _ in range(opts.batch_size - 1):
        image_batch.append(np.array(Image.open("images/trunk.jpg").convert("RGB")))
        pts_batch.append(np.array([[[500, 375]], [[650, 750]]]))
        labels_batch.append(np.array([[1], [1]]))

    # Run phase
    masks_batch, scores_batch, logits_batch = run(
        set_image, predictor, image_batch, pts_batch, labels_batch
    )

    # Save segmented images
    save_segmented_image(image_batch, pts_batch, masks_batch, labels_batch, scores_batch)


if __name__ == "__main__":
    main()
