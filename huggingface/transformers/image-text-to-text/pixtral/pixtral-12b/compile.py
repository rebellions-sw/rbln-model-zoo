import os

from optimum.rbln import RBLNAutoModelForVision2Seq


def main():
    model_id = "mistral-community/pixtral-12b"
    model = RBLNAutoModelForVision2Seq.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "vision_tower": {
                "batch_size": 1,
                "output_hidden_states": True,
            },
            "language_model": {
                "tensor_parallel_size": 8,
                "use_inputs_embeds": True,
                "batch_size": 1,
                "max_seq_len": 131_072,
                "kvcache_partition_len": 16_384,
            },
        },
    )
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
