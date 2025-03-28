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
- General Requirements: Rebellions Compiler

    The `rebel-compiler` Python package is required for all workflows involing RBLN NPUs. Please install it before processing. You need an [RBLN portal account](https://docs.rbln.ai/getting_started/installation_guide.html#installation-guide) to install `rebel-compiler`.
    ```bash
    pip3 install -i https://pypi.rbln.ai/simple rebel-compiler
    ```

- HuggingFace Models
  
    [Optimum RBLN](https://docs.rbln.ai/software/optimum/optimum_rbln.html) serves as a bridge connecting the HuggingFace `transformers`/`diffusers` libraries to RBLN NPUs. It offers a set of tools that enable easy model compilation and inference for both single and multi-NPU (Rebellions Scalable Design) configurations, across a range of downstream tasks. To install prereuisites for HuggingFace models, navigate to the model's directory and use its requirements.txt:
    ```bash
    pip3 install -r <model_directory>/requirements.txt
    ```
    For instance:
    ```bash
    pip3 install -r huggingface/transformers/question-answering/bert/requirements.txt
    ```

- PyTorch

    Each PyTorch model now includes its own `requirements.txt`. Install the prerequisites for your specific model by navigtating to the relevant directory:
    ```bash
    pip3 install -r pytorch/<model_directory>/requirements.txt 
    ```

- TensorFlow

    Similarly, TensorFlow models provide a `requirements.txt` in their respective directories. Install prerequisites as follows:
    ```bash
    pip3 install -r tensorflow/<model_directory>/requirements.txt
    ```

- Language Binding
    - C/C++

        - The C/C++ API can be installed via the APT repository. Please refer to [C/C++ binding Installation Guide](https://docs.rbln.ai/software/api/language_binding/c/installation.html) for more details.

## Model List
You can find the complete list of models on our [homepage](https://rebellions.ai/developers/model-zoo) and in the [online documentation](https://docs.rbln.ai/misc/pytorch_modelzoo.html). 

## Developer Resources
Explore [RBLN SDK documentation](https://docs.rbln.ai) to access detailed information including:

- Tutorials
    - [PyTorch: ResNet50](https://docs.rbln.ai/software/api/python/tutorial/basic/pytorch_resnet50.html)
    - [TensorFlow: BERT-base](https://docs.rbln.ai/software/api/python/tutorial/basic/tensorflow_bert.html)
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
- [Model Serving Guide using Nvidia Triton Inference Server](https://docs.rbln.ai/software/model_serving/nvidia_triton_inference_server/installation.html)
- [vLLM Support](https://docs.rbln.ai/software/model_serving/vllm_support/vllm-rbln.html)
- [Device Management](https://docs.rbln.ai/software/system_management/device_management.html)

## Release Notes
For detailed information on updates and changes, please refer to the [release notes](CHANGELOG.md).

## Getting Help
If you encounter any problem with the examples provided, please open an issue on GitHub. Our team will assist you as soon as possible.
