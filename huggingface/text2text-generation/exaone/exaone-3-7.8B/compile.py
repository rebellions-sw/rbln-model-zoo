import os

from optimum.rbln import RBLNExaoneForCausalLM


def main():
    model_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"

    # Compile and export
    model = RBLNExaoneForCausalLM.from_pretrained(
        model_id=model_id,
        export=True,  # export a PyTorch model to RBLN model with optimum
        rbln_batch_size=1,
        rbln_max_seq_len=4096,  # default "max_position_embeddings"
        rbln_tensor_parallel_size=4,
        trust_remote_code=True,
    )

    # Save compiled results to disk
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
