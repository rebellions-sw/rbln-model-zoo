import argparse
import os
import numpy as np
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from transformers import pipeline
from optimum.rbln import RBLNStableDiffusionControlNetImg2ImgPipeline


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
        default="lego batman and robin",
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"runwayml/{args.model_name}"
    prompt = args.prompt

    # Load compiled model
    pipe = RBLNStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        scheduler=UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler"),
    )

    # Prepare inputs
    image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"
    )

    def get_depth_map(image, depth_estimator):
        image = depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        detected_map = torch.from_numpy(image).float() / 255.0
        depth_map = detected_map.permute(2, 0, 1)
        return depth_map

    depth_estimator = pipeline("depth-estimation")
    depth_map = get_depth_map(image, depth_estimator).unsqueeze(0)

    # Generate image
    new_image = pipe(prompt=prompt, image=image, control_image=depth_map).images[0]

    # Save image result
    new_image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
