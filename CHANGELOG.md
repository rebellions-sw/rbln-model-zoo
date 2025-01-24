# Change Log

## Feburary, 4th 2025 (v0.5.6)
- Compatible version:
    - `rebel-compiler`: v0.7.1
    - `optimum-rbln`: v0.2.0
- Added new models:
    - `Natural Language Processing`:
        - Llama3.1-8B/70B
        - Llama3.2-3B
        - Llama3.3-70B
        - KONI-Llama3.1-8B
    - `Computer Vision`
        - YOLOv10-n/s/m/b/l/x

## December, 27th 2024 (v0.5.5)
- Compatible version:
    - `rebel-compiler`: v0.6.2
    - `optimum-rbln`: v0.1.15
- Added new models:
    - `Natural Language Processing`:
        - EXAONE-3.5-2.4b
        - EXAONE-3.5-7.8b
    - `Generative AI`
        - Stable Diffusion
            - Inpainting
        - Stable Diffusion XL
            - Inpainting
            - Text to Image + ControlNet
            - Image to Image + ControlNet
        - Stable Diffusion V3
            - Text to Image
            - Image to Image
            - Inpainting
    - `Computer Vision`
        - YOLOv5-Face
    - `vLLM`
        - Llama3
        - BART
        - Llava-Next
- Improved the formatting of all model code for better readability and maintainability
- Added supplementary guides for the model serving tutorials, [Tutorial > Advanced > LLM Serving > LLM Serving with Continous Batching](https://docs.rbln.ai/tutorial/advanced/llm_serving_vllm.html) and [Software > Model Serving > Nvidia Triton Infernece Server](https://docs.rbln.ai/software/model_serving/tritonserver.html)

## November, 27th 2024 (v0.5.4)
- Compatible version:
    - `rebel-compiler`: v0.6.1
    - `optimum-rbln`: v0.1.13
- Added new models:
    - `Natural Language Processing`:
        - Qwen2.5-7b
        - Qwen2.5-14b
        - LLama3-8b + LoRA
    - `Generative AI`
        - StableDiffusion + LCM LoRA
        - StableDiffusionXL + LCM LoRA + Pixel LoRA
- Dependency updates:
    - PyTorch: udpated to version `v2.5.1`
    - TensorFlow: updated to version `v2.18.0`
        - TensorFlow has adopted Keras 3.0 starting from version `v2.16.0`. To ensure compatibility with existing models using Keras 2, we have incorporated the `tf_keras` package into the existing TensorFlow codebase. This allows seamless integration and maintains backward compatibility with the legacy models 
- Model deprecation:
    - 3DDFA  

## October, 30th 2024 (v0.5.3)
- Compatible version:
    - `rebel-compiler`: v0.5.12
    - `optimum-rbln`: v0.1.12
- Added new models:
    - `Natural Language Processing`:
        - Qwen2-7b
        - EXAONE-3.0-7.8b
        - Salamandra-7b
        - Phi-2
        - Whisper-large-v3-turbo
    - `Computer Vision`
        - ViT-large
        - SAM2.1_hiera_large 

## September, 27th 2024 (v0.5.2)
- Compatible version:
    - `rebel-compiler`: v0.5.10
    - `optimum-rbln`: v0.1.11
- Added new models:
    - `Multi Modal`:
        - Llava-Next
    - `Natural Language Processing`:
        - E5-Base-4k
        - KoBART
        - BGE-Reranker-Base/Large
    - `Computer Vision`
        - MotionBERT Action Recognition
- Updated Whisper models to support generating token timestamps and long-form transcription

## August, 30th 2024 (v0.5.1)
- Compatible version:
    - `rebel-compiler`: v0.5.9
    - `optimum-rbln`: v0.1.9
- Added new models:
    - `Natural Language Processing`:
        - Gemma-2B/7B
        - Mistral-7B
        - DistilBERT
    - `Computer Vision`
        - MotionBERT Pose Estimation
        - pytorch_dynamo
            - YOLOv3/4/5/6/X

## August, 16th 2024 (v0.5.0)
- Compatible version:
    - `rebel-compiler`: v0.5.8
    - `optimum-rbln`: v0.1.8
- We are excited to announce support for pytorch2.0 dynamic feature, i.e. torch.compile(). Sample code can be found in `rbln-model-zoo/pytorch_dynamo`.
- Added new models:
    - `Natural Language Processing`:
        - Mi:dm-7b
        - BGE-M3
        - BGE-Reranker-v2-m3
        - SecureBERT
        - RoBERTa

## July, 25th 2024 (v0.4.1)
- Compatible version:
    - `rebel-compiler`: v0.5.7
    - `optimum-rbln`: v0.1.7
- Added a new model:
    - `Computer Vision`:
        - DPT-large

## July, 10th 2024 (v0.4.0)
- Compatible version:
    - `rebel-compiler`: v0.5.2
    - `optimum-rbln`: v0.1.4
- Added the following models:
    - `Computer Vision`:
        - `Image Classification`: EfficientNet-b0/1/2/3/4/5/6/7, EfficientNetV2-b0/b1/b2/b3/small/medium/large, WideResNet50_2/101_2, MnasNet0_5/0_75/1_0/1_3, NasNetMobile/Large, MobileNet, MobileNetV2, MobileNetV3-small/large, ResNet18/34/50/101/152, ResNetV2-50/101/152, AlexNet, VGG11/13/16/19, VGG_BN_11/13/16/19, SqueezeNet1_0/1_1, GoogLeNet, InceptionV3, ShuffleNetV2_x0_5/x1_0/x1_5/x2_0, DenseNet121/161/169/201, InceptionResNetV2, Xception, ResNetRS50/101/152/200/270/350/420, ResNext50_32x4d/101_32x8d/101_64x4d, ConvNext-tiny/small/base/large/xlarge, RegNetY-200MF/400MF/600MF/800MF/1.6GF/3.2GF/4GF/6.4GF/8GF/12GF/16GF/32GF/128GF, RegNetX-200MF/400MF/600MF/800MF/1.6GF/3.2GF/4GF/6.4GF/8GF/12GF/16GF/32GF, 
        - `Object Detection`: YOLOv3/v4/v5/v6/v7/v8/X
        - `Face Alignment`: 3DDFA_V2
        - `Segmentation`: FCN_ResNet50/101, DeepLabV3_ResNet50/101, DeepLabV3_MobileNetV3Large, UNet
        - `Video Classification`: S3D, R3D_18, MC3_18, R(2+1)D_18
        - `Depth Estimation`: UNet-DenseNet201
    - `Natural Language Processing`:
        - `Large Language Model`: LLama3-8b, Llama2-7b/13b, Solar-10.7b, EEVE-Korean-10.8b, T5-small/base/large/3b, BART-base/large, GPT2-base/medium/large/xlarge
        - `Question & Answering`: BERT-base/large
        - `Masked Language Model`: BERT-base/large
    - `Speech Processing`:
        - `Speech Recognition`: Wav2Vec2, Whisper-Tiny/Small/Base/Large
        - `Audio Classification`: Audio-Spectogram-Transformer
        - `Speech Seperation`: ConvTasNet
    - `Generative AI`:
        - `Image Generation`: StableDiffusionV1.5, Controlnet, StableDiffusionXL-Base-1.0, SDXL-Turbo
