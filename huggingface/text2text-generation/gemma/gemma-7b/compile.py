import os

from optimum.rbln import RBLNGemmaForCausalLM


def main():
    model_id = "google/gemma-7b-it"

    # Compile and export
    model = RBLNGemmaForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=8192,  # default "max_position_embeddings"
        rbln_tensor_parallel_size=4,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
