import argparse
import os
import cv2
import numpy as np
from PIL import Image

from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from optimum.rbln import RBLNStableDiffusionControlNetPipeline


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["stable-diffusion-v1-5"],
        default="stable-diffusion-v1-5",
        help="(str) model type, diffusers stable diffusion model name.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="the mona lisa",
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"runwayml/{args.model_name}"
    prompt = args.prompt

    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )

    # input image preprocessing
    np_image = np.array(image)
    np_image = cv2.Canny(np_image, 100, 200)
    np_image = np_image[:, :, None]
    np_image = np.concatenate([np_image, np_image, np_image], axis=2)
    canny_image = Image.fromarray(np_image)

    # Load compiled model
    pipe = RBLNStableDiffusionControlNetPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        scheduler=UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler"),
    )

    # Generate image
    image = pipe(prompt=prompt, image=canny_image).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
