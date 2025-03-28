import os

from optimum.rbln import RBLNLlamaForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = RBLNLlamaForCausalLM.from_pretrained(
    model_id=model_id,
    export=True,
    rbln_batch_size=1,
    rbln_max_seq_len=8192,
    rbln_tensor_parallel_size=4,
)

# Save the compiled model
model.save_pretrained(os.path.basename(model_id))
