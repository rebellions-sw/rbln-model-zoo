import argparse

import rebel
from postprocess import postprocessing
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

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime("bge-m3-compute-score.rbln", tensor_type="pt")

    # Set `max sequence length` of the compiled model
    MAX_SEQ_LEN = 8192

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    input_q = tokenizer(
        args.query, padding="max_length", return_tensors="pt", max_length=MAX_SEQ_LEN
    )
    input_m = tokenizer(
        args.message, padding="max_length", return_tensors="pt", max_length=MAX_SEQ_LEN
    )

    # run model
    q_output = module.run(input_q.input_ids, input_q.attention_mask)
    m_output = module.run(input_m.input_ids, input_m.attention_mask)

    # Get similarity score
    scores = postprocessing(
        q_vecs=q_output,
        p_vecs=m_output,
        query_input=input_q,
        batch_size=1,
        weights_for_different_modes=[0.4, 0.2, 0.4],
    )

    # Show text and result
    print("--- query ---")
    print(args.query)
    print("--- message ---")
    print(args.message)
    print("--- score ---")
    print(scores)


if __name__ == "__main__":
    main()
