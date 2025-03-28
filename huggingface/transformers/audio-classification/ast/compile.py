import os

from optimum.rbln import RBLNASTForAudioClassification


def main():
    model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"

    # Compile and export
    model = RBLNASTForAudioClassification.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
