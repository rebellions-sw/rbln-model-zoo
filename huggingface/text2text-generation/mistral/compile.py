import os

from optimum.rbln import RBLNMistralForCausalLM


def main():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    # Compile and export
    model = RBLNMistralForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=32768,  # default "max_position_embeddings"
        rbln_tensor_parallel_size=8,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
