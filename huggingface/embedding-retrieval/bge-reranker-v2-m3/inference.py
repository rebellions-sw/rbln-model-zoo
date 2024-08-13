import os
import argparse


from transformers import AutoTokenizer
from optimum.rbln import RBLNXLMRobertaForSequenceClassification


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default="what is panda?",
        help="(str) type, query context for score",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="The giant panda (Ailuropoda melanoleuca), "
        "sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        help="(str) type, messege context for score",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "BAAI/bge-reranker-v2-m3"

    # Load compiled model
    model = RBLNXLMRobertaForSequenceClassification.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(
        args.query, args.message, padding="max_length", return_tensors="pt", max_length=8192
    )

    # run model
    output = model(inputs.input_ids, inputs.attention_mask)

    # get score
    score = output.view(-1).float()

    # Show text and result
    print("--- query ---")
    print(args.query)
    print("--- message ---")
    print(args.message)
    print("--- score ---")
    print(score)


if __name__ == "__main__":
    main()
