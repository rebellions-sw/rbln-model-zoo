import argparse
import os
from optimum.rbln import RBLNStableDiffusionXLPipeline


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["sdxl-turbo"],
        default="sdxl-turbo",
        help="(str) model type, diffusers stable diffusion xl model name.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
        help="(str) type, prompt for generate image",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="guidance_scale for sdxl-turbo",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"stabilityai/{args.model_name}"
    prompt = args.prompt
    guidance_scale = args.guidance_scale

    # Load compiled model
    pipe = RBLNStableDiffusionXLPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate image
    image = pipe(prompt, num_inference_steps=1, guidance_scale=guidance_scale).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
