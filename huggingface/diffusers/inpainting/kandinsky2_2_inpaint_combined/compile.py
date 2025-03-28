import os

from optimum.rbln import RBLNKandinskyV22InpaintCombinedPipeline


def main():
    model_id = "kandinsky-community/kandinsky-2-2-decoder-inpaint"

    # Compile and export
    pipe = RBLNKandinskyV22InpaintCombinedPipeline.from_pretrained(
        model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_config={
            "img_height": 768,
            "img_width": 768,
        },
    )

    # Save compiled results to disk
    pipe.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
