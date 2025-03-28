import argparse
import os

from optimum.rbln import RBLNStableDiffusionXLInpaintPipeline


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="guidance_scale for sdxl-turbo",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    guidance_scale = args.guidance_scale

    # Compile and export
    pipe = RBLNStableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_guidance_scale=guidance_scale,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
