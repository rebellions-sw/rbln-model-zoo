import argparse
import os

from diffusers.utils import load_image
from optimum.rbln import RBLNStableDiffusion3InpaintPipeline


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
    pipe = RBLNStableDiffusion3InpaintPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config={
            "text_encoder": {"device": 1},
            "text_encoder_2": {"device": 1},
            "text_encoder_3": {"device": 0},
            "transformer": {"device": 1},
            "vae": {"device": 1},
        },
    )

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
    source = load_image(img_url)
    mask = load_image(mask_url)

    # Generate image
    image = pipe(prompt, image=source, mask_image=mask, guidance_scale=7.5).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
