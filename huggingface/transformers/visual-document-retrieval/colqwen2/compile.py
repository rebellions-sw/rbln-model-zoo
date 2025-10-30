import os

from optimum.rbln import RBLNColQwen2ForRetrieval


def main():
    model_id = "vidore/colqwen2-v1.0-hf"

    # Compile and export the model
    model = RBLNColQwen2ForRetrieval.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "visual": {
                "max_seq_lens": 6400,
                "device": 0,
            },
            "tensor_parallel_size": 4,
            "max_seq_len": 32_768,
            "device": [0, 1, 2, 3],
        },
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
