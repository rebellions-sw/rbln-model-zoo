# Change Log

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
