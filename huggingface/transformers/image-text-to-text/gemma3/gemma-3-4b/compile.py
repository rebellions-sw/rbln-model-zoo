import os

from optimum.rbln import RBLNGemma3ForConditionalGeneration


def main():
    model_id = "google/gemma-3-4b-it"
    model_dir = os.path.basename(model_id)

    model = RBLNGemma3ForConditionalGeneration.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "language_model": {
                "tensor_parallel_size": 8,
                "kvcache_partition_len": 16_384,
                "use_inputs_embeds": True,
            }
        },
    )
    model.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
