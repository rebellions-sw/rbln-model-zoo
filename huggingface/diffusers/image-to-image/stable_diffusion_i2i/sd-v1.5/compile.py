import argparse
import os

from optimum.rbln import RBLNStableDiffusionImg2ImgPipeline


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_height",
        type=int,
        default=512,
        help="input image width for generation",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=768,
        help="input image height for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "benjamin-paine/stable-diffusion-v1-5"

    # Compile and export
    pipe = RBLNStableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_img_width=args.img_width,
        rbln_img_height=args.img_height,
        rbln_config={
            "img_width": args.img_width,
            "img_height": args.img_height,
            "unet": {"batch_size": 2},
        },
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
