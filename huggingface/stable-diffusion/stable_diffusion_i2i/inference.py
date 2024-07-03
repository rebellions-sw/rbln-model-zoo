import argparse
import os
from io import BytesIO
import requests
from PIL import Image
from optimum.rbln import RBLNStableDiffusionImg2ImgPipeline


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
        default="A fantasy landscape, trending on artstation",
        help="(str) type, prompt for generate image",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=512,
        help="input image width for generation",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=768,
        help="input image height for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"runwayml/{args.model_name}"
    prompt = args.prompt

    # Load compiled model
    pipe = RBLNStableDiffusionImg2ImgPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((args.img_width, args.img_height))

    # Generate image
    image = pipe(prompt, init_image, strength=0.75, guidance_scale=7.5).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
