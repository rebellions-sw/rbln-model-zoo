import os

from optimum.rbln import RBLNRobertaForSequenceClassification


def main():
    model_id = "cardiffnlp/twitter-roberta-base-emotion"

    # Compile and export
    model = RBLNRobertaForSequenceClassification.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=512,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
