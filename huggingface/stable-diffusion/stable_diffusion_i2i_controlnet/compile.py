import argparse
import os

from diffusers import ControlNetModel
from optimum.rbln import RBLNDPTForDepthEstimation, RBLNStableDiffusionControlNetImg2ImgPipeline


def parsing_argument():
    parser = argparse.ArgumentParser()
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
    model_id = "benjamin-paine/stable-diffusion-v1-5"

    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth")

    # Compile and export
    pipe = RBLNStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_img_width=args.img_width,
        rbln_img_height=args.img_height,
        controlnet=controlnet,
    )
    dpt = RBLNDPTForDepthEstimation.from_pretrained(model_id="Intel/dpt-large", export=True)

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))
    dpt.save_pretrained(os.path.basename("dpt-large"))


if __name__ == "__main__":
    main()
