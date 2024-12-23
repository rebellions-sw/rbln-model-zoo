import os

from optimum.rbln import RBLNStableDiffusion3Pipeline


def main():
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    # Compile and export
    pipe = RBLNStableDiffusion3Pipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_img_height=1024,
        rbln_img_width=1024,
        rbln_guidance_scale=7.0,
        rbln_config={
            "text_encoder": {"device": 1},
            "text_encoder_2": {"device": 1},
            "text_encoder_3": {"device": 0},
            "transformer": {"device": 1},
            "vae": {"device": 1},
        },
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
