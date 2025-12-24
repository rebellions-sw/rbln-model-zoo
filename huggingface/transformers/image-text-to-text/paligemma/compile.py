import os

from optimum.rbln import RBLNAutoModelForVision2Seq


def main():
    model_id = "google/paligemma-3b-mix-224"
    model = RBLNAutoModelForVision2Seq.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "language_model": {
                "batch_size": 1,
                "max_seq_len": 8192,  # default "max_position_embeddings"
                "tensor_parallel_size": 4,
                "prefill_chunk_size": 8192,
            },
        },
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
