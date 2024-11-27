import os

from optimum.rbln import RBLNStableDiffusionPipeline


def main():
    model_id = "runwayml/stable-diffusion-v1-5"

    # Compile and export
    pipe = RBLNStableDiffusionPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
