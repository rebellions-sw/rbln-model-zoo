import os

from optimum.rbln import RBLNLlamaForCausalLM


def main():
    model_id = "meta-llama/Llama-2-13b-chat-hf"

    # Compile and export
    model = RBLNLlamaForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=4096,  # default "max_position_embeddings"
        rbln_tensor_parallel_size=8,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
