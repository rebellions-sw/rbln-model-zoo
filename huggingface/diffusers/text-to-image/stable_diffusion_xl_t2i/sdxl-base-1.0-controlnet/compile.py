import argparse
import os

from diffusers import AutoencoderKL, ControlNetModel
from optimum.rbln import RBLNStableDiffusionXLControlNetPipeline


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_width",
        type=int,
        default=1024,
        help="input image width for generation",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=1024,
        help="input image height for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")

    # Compile and export
    pipe = RBLNStableDiffusionXLControlNetPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        controlnet=controlnet,
        vae=vae,
        rbln_img_width=args.img_width,
        rbln_img_height=args.img_height,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
