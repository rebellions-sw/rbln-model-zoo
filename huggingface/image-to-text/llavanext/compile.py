import os

from optimum.rbln import RBLNLlavaNextForConditionalGeneration


def main():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    model = RBLNLlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        export=True,
        rbln_config={
            "language_model": {
                "tensor_parallel_size": 8,
                "use_inputs_embeds": True,
            }
        },
    )
    model.save_pretarined(os.path.basename(model_id))


if __name__ == "__main__":
    main()
