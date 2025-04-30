import argparse
import os

import torch
from optimum.rbln import RBLNXLMRobertaModel
from transformers import AutoTokenizer


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default="what is panda?",
        help="(str) type, input query context",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="The giant panda (Ailuropoda melanoleuca), "
        "sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        help="(str) type, input messege context",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = "BAAI/bge-m3"

    # Load compiled model
    model = RBLNXLMRobertaModel.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_q = tokenizer(args.query, padding="max_length", return_tensors="pt", max_length=8192)
    input_m = tokenizer(args.message, padding="max_length", return_tensors="pt", max_length=8192)

    # run model
    q_output = model(input_q.input_ids, input_q.attention_mask)[0]
    m_output = model(input_m.input_ids, input_m.attention_mask)[0]
    q_output = torch.nn.functional.normalize(q_output[0][:, 0], dim=-1)
    m_output = torch.nn.functional.normalize(m_output[0][:, 0], dim=-1)

    # get similarity score
    score = q_output @ m_output.T

    # Show text and result
    print("--- query ---")
    print(args.query)
    print("--- message ---")
    print(args.message)
    print("--- score ---")
    print(score)


if __name__ == "__main__":
    main()
