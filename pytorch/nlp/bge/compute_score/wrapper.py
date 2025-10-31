import torch


class RBLNBGEM3ComputeScoreWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super(RBLNBGEM3ComputeScoreWrapper, self).__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        hidden_state = self.model.model(**inputs, return_dict=True).last_hidden_state

        # dense_embedding
        dense_vecs = torch.nn.functional.normalize(hidden_state[:, 0], dim=-1)

        # sparse_embedding
        token_weights = self.model.sparse_embedding(
            hidden_state, input_ids, return_embedding=False
        )
        sparse_vecs = torch.zeros(
            input_ids.size(0),
            input_ids.size(1),
            self.model.vocab_size,
            dtype=token_weights.dtype,
            device=token_weights.device,
        )
        sparse_vecs = torch.scatter(
            sparse_vecs,
            dim=sparse_vecs.dim() - 1,
            index=input_ids.unsqueeze(-1),
            src=token_weights,
        )
        sparse_vecs = torch.max(sparse_vecs, dim=1).values
        unused_tokens = [
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        ]
        for token_id in unused_tokens:
            sparse_vecs = torch.concat(
                [
                    sparse_vecs[:, :token_id],
                    torch.zeros([sparse_vecs.shape[0], 1], dtype=torch.float32),
                    sparse_vecs[:, token_id + 1 :],
                ],
                dim=1,
            )

        # colbert_embedding
        colbert_vecs = self.model.colbert_embedding(hidden_state, attention_mask)
        colbert_vecs = torch.nn.functional.normalize(colbert_vecs, dim=-1)

        return (dense_vecs, sparse_vecs, colbert_vecs)
