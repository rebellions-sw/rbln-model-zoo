import os

from optimum.rbln import RBLNStableDiffusionInpaintPipeline


def main():
    model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"

    # Compile and export
    pipe = RBLNStableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_guidance_scale=7.5,
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
