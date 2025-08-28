import os

from optimum.rbln import RBLNAutoModelForTextEncoding


def main():
    model_id = "BAAI/bge-m3"

    # Compile and export
    model = RBLNAutoModelForTextEncoding.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=8192,  # default "max_position_embeddings"
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
