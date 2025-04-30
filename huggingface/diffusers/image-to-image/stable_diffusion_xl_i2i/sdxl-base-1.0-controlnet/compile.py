import argparse
import os

from diffusers import AutoencoderKL, ControlNetModel
from optimum.rbln import RBLNDPTForDepthEstimation, RBLNStableDiffusionXLControlNetImg2ImgPipeline


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

    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0-small")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")

    # Compile and export
    pipe = RBLNStableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        controlnet=controlnet,
        vae=vae,
        rbln_img_width=args.img_width,
        rbln_img_height=args.img_height,
        rbln_config={
            "img_width": args.img_width,
            "img_height": args.img_height,
            "unet": {"batch_size": 2},
        },
    )
    dpt = RBLNDPTForDepthEstimation.from_pretrained(model_id="Intel/dpt-hybrid-midas", export=True)

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))
    dpt.save_pretrained(os.path.basename("dpt-hybrid-midas"))


if __name__ == "__main__":
    main()
