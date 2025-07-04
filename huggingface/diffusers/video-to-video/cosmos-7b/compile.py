import os

from optimum.rbln import RBLNCosmosVideoToWorldPipeline


def main():
    model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Video2World"

    # By default, the generated video is a 4-second clip with a resolution of 704x1280 pixels and a frame rate of 30 frames per second (fps).
    # The video content visualizes the input text description as a short animated scene, capturing key elements within the specified time constraints.
    # Aspect ratios and resolutions are configurable, with options including 1:1 (960x960 pixels), 4:3 (960x704 pixels), 3:4 (704x960 pixels),
    # 16:9 (1280x704 pixels), and 9:16 (704x1280 pixels).
    height = 704
    width = 1280

    # Compile and export
    pipe = RBLNCosmosVideoToWorldPipeline.from_pretrained(
        model_id,
        export=True,  # export PyTorch models to RBLN models with optimum
        rbln_config={
            "height": height,
            "width": width,
            "create_runtimes": False,
            "transformer": {
                "tensor_parallel_size": 4,
            },
        },
    )

    # Save all compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
