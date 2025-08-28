import argparse
import os

from optimum.rbln import RBLNAutoModelForQuestionAnswering
from transformers import pipeline


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        type=str,
        default="Who was Jim Henson?",
        help="(str) type, question text",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="Jim Henson was a nice puppet",
        help="(str) type, context text",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "distilbert/distilbert-base-uncased-distilled-squad"

    # Load compiled model
    model = RBLNAutoModelForQuestionAnswering.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Generate Answer
    pipe = pipeline(
        "question-answering",
        model=model,
        tokenizer=model_id,
        padding="max_length",
        max_seq_len=512,
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
