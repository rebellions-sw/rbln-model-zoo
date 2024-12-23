import os

from optimum.rbln import RBLNDistilBertForQuestionAnswering


def main():
    model_id = "distilbert/distilbert-base-uncased-distilled-squad"

    # Compile and export
    model = RBLNDistilBertForQuestionAnswering.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
    )
    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
