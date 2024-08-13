import torch

from FlagEmbedding.BGE_M3.modeling import BGEM3ForInference
import rebel  # RBLN Compiler


class RBLNBGEM3DenseEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model):
        super(RBLNBGEM3DenseEmbeddingWrapper, self).__init__()
        self.model = model.model

    def forward(self, input_ids, attention_mask):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        logit = self.model(**inputs, return_dict=True).last_hidden_state
        dense_vecs = torch.nn.functional.normalize(logit[:, 0], dim=-1)
        return dense_vecs


def main():
    model_id = "BAAI/bge-m3"

    # Compile and export
    model = BGEM3ForInference(
        model_name=model_id,
        normlized=True,
        sentence_pooling_method="cls",
    )
    model = RBLNBGEM3DenseEmbeddingWrapper(model).eval()

    # Set `max sequence length` of the compiled model
    MAX_SEQ_LEN = 8192

    # Compile
    input_info = [
        ("input_ids", [1, MAX_SEQ_LEN], "int64"),
        ("attention_mask", [1, MAX_SEQ_LEN], "int64"),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save("bge-m3-dense-embedding.rbln")


if __name__ == "__main__":
    main()
