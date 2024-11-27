import os
import argparse

from transformers import pipeline
from optimum.rbln import RBLNBertForQuestionAnswering


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        choices=["base", "large"],
        default="base",
        help="(str) type, Size of BERT. [base or large]",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is Rebellions?",
        help="(str) type, question text",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="Rebellions is the best NPU company.",
        help="(str) type, context text",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = (
        "deepset/bert-base-cased-squad2"
        if args.model_name == "base"
        else "deepset/bert-large-uncased-whole-word-masking-squad2"
    )

    # Load compiled model
    model = RBLNBertForQuestionAnswering.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate Answer
    pipe = pipeline(
        "question-answering",
        model=model,
        tokenizer=model_id,
        padding="max_length",
        max_seq_len=512,  # default "max_position_embedding"
    )
    answer = pipe(question=args.question, context=args.context)

    # Result
    print("--- question ---")
    print(args.question)
    print("--- context ---")
    print(args.context)
    print("--- Result ---")
    print(answer)


if __name__ == "__main__":
    main()