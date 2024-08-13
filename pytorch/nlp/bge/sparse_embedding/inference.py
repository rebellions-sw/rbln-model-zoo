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
from collections import defaultdict

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


# ref: https://github.com/FlagOpen/FlagEmbedding/blob/502562bcf1af315160db49c0d2b13b1d371f3352/FlagEmbedding/bge_m3.py#L116
def process_token_weights(token_weights, input_ids, tokenizer):
    # conver to dict
    result = defaultdict(int)
    unused_tokens = set(
        [
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        ]
    )
    # token_weights = np.ceil(token_weights * 100)
    for w, idx in zip(token_weights, input_ids):
        if idx not in unused_tokens and w > 0:
            idx = str(idx)
            # w = int(w)
            if w > result[idx]:
                result[idx] = w
    return result


# ref: https://github.com/FlagOpen/FlagEmbedding/blob/502562bcf1af315160db49c0d2b13b1d371f3352/FlagEmbedding/bge_m3.py#L83
def compute_lexical_matching_score(lexical_weights_1, lexical_weights_2):
    scores = 0
    for token, weight in lexical_weights_1.items():
        if token in lexical_weights_2:
            scores += weight * lexical_weights_2[token]
    return scores


def postprocessing(token_weights: list, input_data: list, tokenizer, batch_size):
    tokenizer = [tokenizer] * batch_size

    first_lexical_weights = []
    first_token_weight = token_weights[0].squeeze(-1)
    first_lexical_weights.extend(
        list(
            map(
                process_token_weights,
                first_token_weight.cpu().numpy(),
                input_data[0].input_ids.cpu().numpy().tolist(),
                tokenizer,
            )
        )
    )

    second_lexical_weights = []
    second_token_weight = token_weights[1].squeeze(-1)
    second_lexical_weights.extend(
        list(
            map(
                process_token_weights,
                second_token_weight.cpu().numpy(),
                input_data[1].input_ids.cpu().numpy().tolist(),
                tokenizer,
            )
        )
    )

    lexical_matching_scores = []
    for i in range(batch_size):
        lexical_matching_scores.append(
            compute_lexical_matching_score(first_lexical_weights[i], second_lexical_weights[i])
        )

    return lexical_matching_scores


def main():
    args = parsing_argument()
    model_id = "BAAI/bge-m3"

    # Load compiled model to RBLN runtime module
    module = rebel.Runtime("bge-m3-sparse-embedding.rbln", tensor_type="pt")

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
        tokenizer=tokenizer,
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
