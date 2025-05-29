import os

from optimum.rbln import RBLNOPTForCausalLM


def main():
    model_id = "facebook/opt-6.7b"

    # Compile and export
    model = RBLNOPTForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=2048,  # default "max_position_embeddings"
        rbln_tensor_parallel_size=2,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
