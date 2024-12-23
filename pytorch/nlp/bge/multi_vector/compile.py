import rebel  # RBLN Compiler
import torch
from FlagEmbedding.BGE_M3.modeling import BGEM3ForInference


class RBLNBGEM3ColbertEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model):
        super(RBLNBGEM3ColbertEmbeddingWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        last_hidden_state = self.model.model(**inputs, return_dict=True).last_hidden_state

        # colbert_embedding
        colbert_embedding = self.model.colbert_embedding(last_hidden_state, attention_mask)
        colbert_embedding = torch.nn.functional.normalize(colbert_embedding, dim=-1)

        return colbert_embedding


def main():
    model_id = "BAAI/bge-m3"

    # Compile and export
    model = BGEM3ForInference(
        model_name=model_id,
        normlized=True,
        sentence_pooling_method="cls",
    )
    model = RBLNBGEM3ColbertEmbeddingWrapper(model).eval()

    # Set `max sequence length` of the compiled model
    MAX_SEQ_LEN = 8192

    # Compile
    input_info = [
        ("input_ids", [1, MAX_SEQ_LEN], "int64"),
        ("attention_mask", [1, MAX_SEQ_LEN], "int64"),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save("bge-m3-colbert-embedding.rbln")


if __name__ == "__main__":
    main()
