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
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"runwayml/{args.model_name}"

    # Compile and export
    pipe = RBLNStableDiffusionPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
