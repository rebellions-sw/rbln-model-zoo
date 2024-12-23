import argparse
import os

from optimum.rbln import RBLNStableDiffusion3Pipeline


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of a cat holding a sign that says hello world",
        help="(str) type, prompt for generate image",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    prompt = args.prompt

    # Load compiled model
    pipe = RBLNStableDiffusion3Pipeline.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
        rbln_config={
            "transformer": {"device": 0},
            "text_encoder_3": {"device": 1},
            "text_encoder": {"device": 2},
            "text_encoder_2": {"device": 2},
            "vae": {"device": 2},
        },
    )

    # Generate image
    image = pipe(
        prompt, num_inference_steps=28, height=1024, width=1024, guidance_scale=7.0
    ).images[0]

    # Save image result
    image.save(f"{prompt}.png")


if __name__ == "__main__":
    main()
