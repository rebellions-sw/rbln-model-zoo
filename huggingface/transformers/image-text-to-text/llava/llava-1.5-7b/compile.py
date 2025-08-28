import os

from optimum.rbln import RBLNAutoModelForVision2Seq


def main():
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = RBLNAutoModelForVision2Seq.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "vision_tower": {"output_hidden_states": True},
            "language_model": {
                "tensor_parallel_size": 4,
                "use_inputs_embeds": True,
            },
        },
    )
    model.save_pretrained(os.path.basename(model_id))


if __name__ == "__main__":
    main()
