import argparse
from transformers import BertTokenizer

import rebel  # RBLN Runtime


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["base", "large"],
        default="base",
        help="(str) type, Size of BERT. [base or large]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = (
        "deepset/bert-base-cased-squad2"
        if args.model_name == "base"
        else "deepset/bert-large-uncased-whole-word-masking-squad2"
    )
    MAX_SEQ_LEN = 384

    # Prepare input text sequence for masked language modeling
    tokenizer = BertTokenizer.from_pretrained(model_name)
    question, text = "What is Rebellions?", "Rebellions is the best NPU company."
    inputs = tokenizer(
        question, text, return_tensors="pt", padding="max_length", max_length=MAX_SEQ_LEN
    )

    # Prepare input tensors as numpy array
    input_ids = inputs["input_ids"].numpy()
    token_type_ids = inputs["token_type_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime(f"bert-{args.model_name}.rbln")

    # Run inference
    out = module.run(input_ids, attention_mask, token_type_ids)

    # Decoding final logit to text
    answer_start_index = out[0].argmax()
    answer_end_index = out[1].argmax()
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    print(tokenizer.decode(predict_answer_tokens))


if __name__ == "__main__":
    main()
