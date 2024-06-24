import argparse
import os
from optimum.rbln import RBLNStableDiffusionControlNetImg2ImgPipeline
from diffusers import ControlNetModel


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
        "--img_width",
        type=int,
        default=768,
        help="input image width for generation",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=768,
        help="input image height for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"runwayml/{args.model_name}"

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth")

    # Compile and export
    pipe = RBLNStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_img_width=args.img_width,
        rbln_img_height=args.img_height,
        controlnet=controlnet,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
