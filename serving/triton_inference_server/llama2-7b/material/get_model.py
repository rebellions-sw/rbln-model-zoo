from optimum.rbln import RBLNLlamaForCausalLM

model_id = "meta-llama/Llama-2-7b-chat-hf"
compiled_model = RBLNLlamaForCausalLM.from_pretrained(
    model_id=model_id,
    export=True,
    rbln_max_seq_len=4096,
    rbln_tensor_parallel_size=4,
    rbln_batch_size=4,
)

compiled_model.save_pretrained("rbln-Llama-2-7b-chat-hf")
