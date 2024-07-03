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
        "--guidance_scale",
        type=float,
        default=0.0,
        help="guidance_scale for sdxl-turbo",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"stabilityai/{args.model_name}"
    guidance_scale = args.guidance_scale

    # Compile and export
    pipe = RBLNStableDiffusionXLPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_guidance_scale=guidance_scale,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
