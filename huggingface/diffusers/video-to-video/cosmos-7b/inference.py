import argparse
import os

from diffusers.utils import export_to_video, load_video
from optimum.rbln import RBLNCosmosVideoToWorldPipeline


def parsing_argument():
    parser = argparse.ArgumentParser(description="Run Cosmos Video2World 7B")
    parser.add_argument(
        "--text",
        type=str,
        default="A dynamic and visually captivating video showcases a sleek, dark-colored SUV driving along a narrow dirt road that runs parallel to a vast, expansive ocean. The setting is a rugged coastal landscape, with the road cutting through dry, golden-brown grass that stretches across rolling hills. The ocean, a deep blue, extends to the horizon, providing a stunning backdrop to the scene. The SUV moves swiftly along the road, kicking up a trail of dust that lingers in the air behind it, emphasizing the speed and power of the vehicle. The camera maintains a steady tracking shot, following the SUV from a slightly elevated angle, which allows for a clear view of both the vehicle and the surrounding scenery. The lighting is natural, suggesting a time of day when the sun is high, casting minimal shadows and highlighting the textures of the grass and the glint of the ocean. The video captures the essence of freedom and adventure, with the SUV navigating the isolated road with ease, suggesting a journey or exploration theme. The consistent motion of the vehicle and the dust trail create a sense of continuity and fluidity throughout the video, making it engaging and immersive.",
        help="(str) type, Text prompt for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Video2World"

    # Load all pipeline compiled model
    pipe = RBLNCosmosVideoToWorldPipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config={
            # The `rbln_config` is a dictionary used to pass configurations for the model and its submodules.
            # The `device` parameter specifies which device should be used for each submodule during runtime.
            #
            # Since Cosmos VideoToWorld consists of multiple submodules, loading all submodules onto a single device may occasionally exceed its memory capacity.
            # Therefore, when creating runtimes for each submodule, devices can be divided and assigned to ensure efficient memory utilization.
            #
            # For example:
            # - Assume each device has a memory capacity of 15.7 GiB (e.g., RBLN-CA12).
            # `text_encoder` (~9.2GB), `transformer` (~9.3GB x 1 device, ~5.8GB x 3 devices),  `VAE encoder` (~6.9GB), `VAE decoder` (~6.6GB)
            # `aegis` (~3.7GB x 4 devices), `siglip_encoder` (~4.5GB), `video_safety_model` (~10.0MB), `face_blur_filter` (~150MB)
            "transformer": {
                "device": [0, 1, 2, 3],
            },
            "text_encoder": {
                "device": 4,
            },
            "vae": {
                "device_map": {"encoder": 5, "decoder": 6},
            },
            "safety_checker": {
                "aegis": {"device": [4, 5, 6, 7]},
                "siglip_encoder": {"device": 7},
                "video_safety_model": {"device": 7},
                "face_blur_filter": {"device": 7},
            },
        },
    )

    print(f"Input prompt: {args.text}")

    video = load_video(
        "https://github.com/nvidia-cosmos/cosmos-predict1/raw/refs/heads/main/assets/diffusion/video2world_input1.mp4"
    )

    # Inference with video & text pair and Generate video from them
    output = pipe(video=video, prompt=args.text).frames[0]
    export_to_video(output, "v2w_7b.mp4", fps=30)


if __name__ == "__main__":
    main()
