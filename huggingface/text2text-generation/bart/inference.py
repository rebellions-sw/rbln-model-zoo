import os
import argparse
from transformers import BartTokenizer
from optimum.rbln import RBLNBartForConditionalGeneration


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["base", "large"],
        default="base",
        help="(str) model type, Size of BART [base, large]",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="UN Chief Says There Is No <mask> in Syria",
        help="(str) type, text for generation",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"facebook/bart-{args.model_name}"

    tokenizer = BartTokenizer.from_pretrained(model_id)

    # Load compiled model
    model = RBLNBartForConditionalGeneration.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    inputs = tokenizer(args.text, return_tensors="pt", padding="max_length")

    generation_kwargs = {}
    generation_kwargs["num_beams"] = 1

    # Generate tokens
    output_sequence = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=model.dec_max_seq_len,
        **generation_kwargs,
    )

    # Show text and result
    print("---- text ----")
    print(args.text)
    print("---- Result ----")
    print(tokenizer.decode(output_sequence[0]))


if __name__ == "__main__":
    main()
