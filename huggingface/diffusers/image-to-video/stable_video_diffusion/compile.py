import argparse
import os

from optimum.rbln import RBLNStableVideoDiffusionPipeline

DEFAULT_FRAMES_AND_DECODE_CHUNK_SIZE = {
    "stable-video-diffusion-img2vid": (14, 7),
    "stable-video-diffusion-img2vid-xt": (25, 5),
}


def parsing_argument():
    parser = argparse.ArgumentParser(
        description="Compile Stable Video Diffusion model to RBLN format"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="stable-video-diffusion-img2vid",
        choices=["stable-video-diffusion-img2vid", "stable-video-diffusion-img2vid-xt"],
        help="Model ID of the Stable Video Diffusion model to compile",
    )
    args = parser.parse_args()
    return args


def main():
    args = parsing_argument()
    model_id = f"stabilityai/{args.model_name}"

    # Compile and export
    pipe = RBLNStableVideoDiffusionPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_height=576,
        rbln_width=1024,
        rbln_num_frames=DEFAULT_FRAMES_AND_DECODE_CHUNK_SIZE[args.model_name][0],
        # To ensure successful model compilation and prevent device OOM,
        # decode_chunk_size is set to a divisor of num_frames.
        rbln_decode_chunk_size=DEFAULT_FRAMES_AND_DECODE_CHUNK_SIZE[args.model_name][1],
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
