# MIT License

# Copyright (c) 2022 staoxiao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import torch

from transformers import AutoTokenizer

import rebel


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


# ref: https://github.com/FlagOpen/FlagEmbedding/blob/502562bcf1af315160db49c0d2b13b1d371f3352/FlagEmbedding/bge_m3.py#L130
def process_colbert_vecs(colbert_vecs, attention_mask: list):
    import numpy as np

    # delte the vectors of padding tokens
    tokens_num = np.sum(attention_mask)
    return colbert_vecs[
        : tokens_num - 1
    ]  # we don't use the embedding of cls, so select tokens_num-1


# ref: https://github.com/FlagOpen/FlagEmbedding/blob/502562bcf1af315160db49c0d2b13b1d371f3352/FlagEmbedding/bge_m3.py#L90
def colbert_score(q_reps, p_reps):
    q_reps, p_reps = torch.from_numpy(q_reps), torch.from_numpy(p_reps)
    token_scores = torch.einsum("in,jn->ij", q_reps, p_reps)
    scores, _ = token_scores.max(-1)
    scores = torch.sum(scores) / q_reps.size(0)
    return scores


def postprocessing(token_weights: list, input_data: list, batch_size):
    first_colbert_vecs = []
    first_colbert_vecs.extend(
        list(
            map(
                process_colbert_vecs,
                token_weights[0].cpu().numpy(),
                input_data[0].attention_mask.cpu().numpy(),
            )
        )
    )

    second_colbert_vecs = []
    second_colbert_vecs.extend(
        list(
            map(
                process_colbert_vecs,
                token_weights[1].cpu().numpy(),
                input_data[1].attention_mask.cpu().numpy(),
            )
        )
    )
    final_colbert_scores = []
    for i in range(batch_size):
        final_colbert_scores.append(
            float(colbert_score(first_colbert_vecs[i], second_colbert_vecs[i]).numpy())
        )

    return final_colbert_scores


def main():
    args = parsing_argument()
    model_id = "BAAI/bge-m3"

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime("bge-m3-colbert-embedding.rbln", tensor_type="pt")

    # Set `max sequence length` of the compiled model
    MAX_SEQ_LEN = 8192

    # Prepare inputs
    tokenizer = AutoTokenizer.from_pretrained(model_id)
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
    score = postprocessing(
        token_weights=[q_output, m_output],
        input_data=[input_q, input_m],
        batch_size=1,
    )

    # Show text and result
    print("--- query ---")
    print(args.query)
    print("--- message ---")
    print(args.message)
    print("--- score ---")
    print(score)


if __name__ == "__main__":
    main()
