import rebel  # RBLN Compiler
from FlagEmbedding.BGE_M3.modeling import BGEM3ForInference
from transformers import AutoTokenizer
from wrapper import RBLNBGEM3ComputeScoreWrapper


def main():
    model_id = "BAAI/bge-m3"

    # Compile and export
    model = BGEM3ForInference(
        model_name=model_id,
        normlized=True,
        sentence_pooling_method="cls",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = RBLNBGEM3ComputeScoreWrapper(model, tokenizer).eval()

    # Set `max sequence length` of the compiled model
    MAX_SEQ_LEN = 8192

    # Compile
    input_info = [
        ("input_ids", [1, MAX_SEQ_LEN], "int64"),
        ("attention_mask", [1, MAX_SEQ_LEN], "int64"),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save("bge-m3-compute-score.rbln")


if __name__ == "__main__":
    main()
