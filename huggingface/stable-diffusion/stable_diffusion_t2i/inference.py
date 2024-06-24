import argparse
import os
from optimum.rbln import RBLNStableDiffusionPipeline


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
        default="a photo of an astronaut riding a horse on mars",
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"runwayml/{args.model_name}"
    prompt = args.prompt

    # Load compiled model
    pipe = RBLNStableDiffusionPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate image
    image = pipe(prompt).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
