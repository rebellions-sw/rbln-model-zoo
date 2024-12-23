import argparse
import os

from diffusers.utils import load_image
from optimum.rbln import RBLNStableDiffusion3Img2ImgPipeline


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",  # noqa: E501
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    prompt = args.prompt

    # Load compiled model
    pipe = RBLNStableDiffusion3Img2ImgPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config={
            "transformer": {"device": 0},
            "text_encoder_3": {"device": 1},
            "text_encoder": {"device": 2},
            "text_encoder_2": {"device": 2},
            "vae": {"device": 2},
        },
    )

    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    init_image = load_image(url).resize((1024, 1024))

    # Generate image
    image = pipe(prompt, image=init_image, strength=0.95, guidance_scale=7.5).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
