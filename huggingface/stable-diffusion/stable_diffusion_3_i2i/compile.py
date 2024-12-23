import os

from optimum.rbln import RBLNStableDiffusion3Img2ImgPipeline


def main():
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    # Compile and export
    pipe = RBLNStableDiffusion3Img2ImgPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_img_height=1024,
        rbln_img_width=1024,
        rbln_guidance_scale=7.5,
        rbln_config={
            "transformer": {"device": 0},
            "text_encoder_3": {"device": 1},
            "text_encoder": {"device": 2},
            "text_encoder_2": {"device": 2},
            "vae": {"device": 2},
        },
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
