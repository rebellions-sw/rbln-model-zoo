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


import torch


# ref: https://github.com/FlagOpen/FlagEmbedding/blob/502562bcf1af315160db49c0d2b13b1d371f3352/FlagEmbedding/BGE_M3/modeling.py#L188
def compute_similarity(q_reps, p_reps):
    if len(p_reps.size()) == 2:
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
    return torch.matmul(q_reps, p_reps.transpose(-2, -1))


# ref: https://github.com/FlagOpen/FlagEmbedding/blob/502562bcf1af315160db49c0d2b13b1d371f3352/FlagEmbedding/BGE_M3/modeling.py#L122
def dense_score(q_reps, p_reps, temperature=1.0):
    scores = compute_similarity(q_reps, p_reps) / temperature
    scores = scores.view(q_reps.size(0), -1)
    return scores


# ref: https://github.com/FlagOpen/FlagEmbedding/blob/502562bcf1af315160db49c0d2b13b1d371f3352/FlagEmbedding/BGE_M3/modeling.py#L127
def sparse_score(q_reps, p_reps, temperature=1.0):
    scores = compute_similarity(q_reps, p_reps) / temperature
    scores = scores.view(q_reps.size(0), -1)
    return scores


# ref: https://github.com/FlagOpen/FlagEmbedding/blob/502562bcf1af315160db49c0d2b13b1d371f3352/FlagEmbedding/BGE_M3/modeling.py#L132
def colbert_score(q_reps, p_reps, q_mask: torch.Tensor, temperature=1.0):
    token_scores = torch.einsum("qin,pjn->qipj", q_reps, p_reps)
    scores, _ = token_scores.max(-1)
    scores = scores.sum(1) / q_mask[:, 1:].sum(-1, keepdim=True)
    scores = scores / temperature
    return scores


# ref: https://github.com/FlagOpen/FlagEmbedding/blob/502562bcf1af315160db49c0d2b13b1d371f3352/FlagEmbedding/bge_m3.py#L188
def postprocessing(
    q_vecs: list,
    p_vecs: list,
    query_input: list,
    batch_size,
    weights_for_different_modes: list = [1.0, 1.0, 1.0],
):
    all_scores = {
        "colbert": [],
        "sparse": [],
        "dense": [],
        "sparse+dense": [],
        "colbert+sparse+dense": [],
    }
    assert len(weights_for_different_modes) == 3
    weight_sum = sum(weights_for_different_modes)
    dense_scores = dense_score(q_vecs[0], p_vecs[0])
    sparse_scores = sparse_score(q_vecs[1], p_vecs[1])
    colbert_scores = colbert_score(
        q_vecs[2], p_vecs[2], q_mask=query_input.attention_mask
    )

    inx = torch.arange(batch_size)
    dense_scores, sparse_scores, colbert_scores = (
        dense_scores[inx, inx].float(),
        sparse_scores[inx, inx].float(),
        colbert_scores[inx, inx].float(),
    )
    all_scores["colbert"].extend(colbert_scores.detach().cpu().numpy().tolist())
    all_scores["sparse"].extend(sparse_scores.detach().cpu().numpy().tolist())
    all_scores["dense"].extend(dense_scores.detach().cpu().numpy().tolist())
    all_scores["sparse+dense"].extend(
        (
            (
                sparse_scores * weights_for_different_modes[1]
                + dense_scores * weights_for_different_modes[0]
            )
            / (weights_for_different_modes[1] + weights_for_different_modes[0])
        )
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )
    all_scores["colbert+sparse+dense"].extend(
        (
            (
                colbert_scores * weights_for_different_modes[2]
                + sparse_scores * weights_for_different_modes[1]
                + dense_scores * weights_for_different_modes[0]
            )
            / weight_sum
        )
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )
    return all_scores
