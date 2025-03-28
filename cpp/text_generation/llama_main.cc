#include "llama_class_example.hpp"

int main() {
  LLamaClass llama_cls;
  // Init Model configuration
  llama_cls.InitConfig();

  // Create Model & Runtime
  llama_cls.Prepare();

  // Init LLamaClass
  llama_cls.Init();

  auto input_ids = Tensor<int64_t>(1, 23);
  assert(LoadBinary<int64_t>(llama_cls.GetIdsPath(), input_ids) == true);

  auto past_cached_length = Tensor<int32_t>();
  llama_cls.PrepareInput(input_ids, past_cached_length);

  // Process of Prefill phase
  llama_cls.DoPrefill();

  // Process of Decode phase
  llama_cls.DoDecode();

  // Generate c_text2text_generation_gen_id.bin
  llama_cls.GenerateBinary();

  // Reset LLamaClass for iteration
  llama_cls.Reset();

  // Deinit LLamaClass
  llama_cls.DeInit();

  return 0;
}
