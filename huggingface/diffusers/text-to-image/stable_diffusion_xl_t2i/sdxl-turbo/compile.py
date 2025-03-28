import os

from optimum.rbln import RBLNStableDiffusionXLPipeline


def main():
    model_id = "stabilityai/sdxl-turbo"

    # Compile and export
    # As SDXL-turbo does not use guidance_scale, we disable them with rbln_guidance_scale=0.0
    pipe = RBLNStableDiffusionXLPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_guidance_scale=0.0,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
