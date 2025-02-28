import os

from optimum.rbln import RBLNLlamaForCausalLM


def main():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    # Compile and export
    model = RBLNLlamaForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=131_072,  # default "max_position_embeddings"
        rbln_tensor_parallel_size=16,
        rbln_attn_impl="flash_attn",
        rbln_kvcache_partition_len=16_384,  # Length of KV cache partitions for flash attention
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
