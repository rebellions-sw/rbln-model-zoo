# RBLN Model Zoo
This repository provides a collection of machine learning model examples for [ATOM](https://rebellions.ai/rebellions-product/atom-2), Rebellions' neural processing unit (NPU). We are continuously enhancing the rbln-model-zoo by adding more models in the following categories:

- Natural Language Processing
- Generative AI
- Speech Processing
- Computer Vision

All deep learning examples in the RBLN Model Zoo include two files: `compile.py` and `inference.py`.
- `compile.py`: compile the model and save the compiled results to local storage
- `inference.py`: load the saved compiled results and perform inference

## Install Prerequisites
- HuggingFace
  
    [Optimum RBLN](https://docs.rbln.ai/software/optimum/optimum_rbln.html) serves as a bridge connecting the HuggingFace `transformers`/`diffusers` libraries to RBLN NPUs. It offers a set of tools that enable easy model compilation and inference for both single and multi-NPU (Rebellions Scalable Design) configurations, across a range of downstream tasks. You need an [RBLN portal account](https://docs.rbln.ai/getting_started/installation_guide.html#installation-guide) to install `optimum-rbln`.
    ```bash
    pip3 install -i https://pypi.rbln.ai/simple optimum-rbln
    pip3 install -r huggingface/requirements.txt
    ```

- PyTorch
    ```bash
    pip3 install -r pytorch/requirements.txt 
    ```

- TensorFlow
    ```bash
    pip3 install -r tensorflow/requirements.txt
    ```

- Language Binding
    - C/C++

        - The C/C++ API can be installed via the APT repository. Please refer to [C/C++ binding Installation Guide](https://docs.rbln.ai/software/api/language_binding/c/installation.html) for more details.

## Model List
You can find the complete list of models on our [homepage](https://rebellions.ai/developers/model-zoo) and in the [online documentation](https://docs.rbln.ai/misc/pytorch_modelzoo.html). 

## Developer Resources
Explore [RBLN SDK documentation](https://docs.rbln.ai) to access detailed information including:

- Tutorials
    - [PyTorch: ResNet50](https://docs.rbln.ai/tutorial/basic/pytorch_resnet50.html)
    - [TensorFlow: BERT-base](https://docs.rbln.ai/tutorial/basic/tensorflow_bert.html)
    - [HuggingFace transformers: LLama2-7b](https://docs.rbln.ai/software/optimum/tutorial/llama_7b.html)
    - [HuggingFace diffusers: SDXL-turbo](https://docs.rbln.ai/software/optimum/tutorial/sdxl_turbo.html)
    - [C/C++ binding: ResNet50](https://docs.rbln.ai/software/api/language_binding/c/tutorial/image_classification.html)
    - [C/C++ binding: Yolov8m](https://docs.rbln.ai/software/api/language_binding/c/tutorial/object_detection.html)
- APIs
    - [Python Compile & Runtime](https://docs.rbln.ai/software/api/python_api.html)
    - [HuggingFace Model API (optimum-rbln)](https://docs.rbln.ai/software/optimum/model_api.html)
    - [C/C++ Binding API](https://docs.rbln.ai/software/api/language_binding/c/api.html)
- [Supported Models](https://docs.rbln.ai/misc/pytorch_modelzoo.html)
- [Supported Operations](https://docs.rbln.ai/misc/supported_ops_pytorch.html)
- [Model Serving Guide using Nvidia Triton Inference Server](https://docs.rbln.ai/software/model_serving/tritonserver.html)
- [vLLM Support](https://docs.rbln.ai/tutorial/advanced/llm_serving.html#continuous-batching-support-with-vllm-rbln)
- [Tools](https://docs.rbln.ai/software/tools.html)

## Release Notes
For detailed information on updates and changes, please refer to the [release notes](CHANGELOG.md).

## Getting Help
If you encounter any problem with the examples provided, please open an issue on GitHub. Our team will assist you as soon as possible.
