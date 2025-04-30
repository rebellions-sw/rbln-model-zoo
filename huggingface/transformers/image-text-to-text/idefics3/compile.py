import os

from optimum.rbln import RBLNIdefics3ForConditionalGeneration


def main():
    model_id = "HuggingFaceM4/Idefics3-8B-Llama3"
    model = RBLNIdefics3ForConditionalGeneration.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            # The `device` parameter specifies the device allocation for each submodule during runtime.
            # As Idefics3 consists of multiple submodules, loading them all onto a single device may exceed its memory capacity, especially as the batch size increases.
            # By distributing submodules across devices, memory usage can be optimized for efficient runtime performance.
            "vision_model": {
                "device": 0,
            },
            "text_model": {
                "batch_size": 1,
                "max_seq_len": 131_072,  # default "max_position_embeddings"
                "tensor_parallel_size": 8,
                "use_inputs_embeds": True,
                "attn_impl": "flash_attn",
                "kvcache_partition_len": 16_384,  # Length of KV cache partitions for flash attention
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
        },
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
