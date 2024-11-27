import argparse
import urllib.request

import rebel  # noqa: F401  # needed to use torch dynamo's "rbln" backend.
import torch
import torchvision.transforms as transforms
from torchvision.io.image import read_image

if torch.__version__ >= "2.5.0":
    torch._dynamo.config.inline_inbuilt_nn_modules = False
    
model_map = {
    "tiny": "deit_tiny_patch16_224",
    "small": "deit_small_patch16_224",
    "base": "deit_base_patch16_224",
    "tiny_distilled": "deit_tiny_distilled_patch16_224",
    "small_distilled": "deit_small_distilled_patch16_224",
    "base_distilled": "deit_base_distilled_patch16_224",
    "base_384": "deit_base_patch16_384",
    "base_distilled_384": "deit_base_distilled_patch16_384",
}


_CHANNEL_MEANS = [123.68, 116.28, 103.53]
_CHANNEL_STDS = [58.395, 57.120, 57.385]


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="base",
        choices=[
            "tiny", "small", "base",
            "tiny_distilled", "small_distilled", "base_distilled",
            "base_384", "base_distilled_384",
        ],
        help="(str) type, DeiT model name. (tiny, small, base, tiny_distilled,"
        " small_distilled, base_distilled, base_384, base_distilled_384)",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name
    resize_size = 384 if "384" in model_name else 256
    input_size = 384 if "384" in model_name else 224

    # Instantiate TorchVision model
    model = torch.hub.load("facebookresearch/deit:main", model_map[model_name], pretrained=True)
    model.eval()

    # Prepare input image
    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    img_path = "./tabby.jpg"
    with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
        f.write(response.read())
    img = read_image(img_path)

    # Preprocessing Image
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.Normalize(_CHANNEL_MEANS, _CHANNEL_STDS),
        ]
    )
    batch = preprocess(img.to(torch.float32)).unsqueeze(0)

    # Compile the model
    model = torch.compile(
        model,
        backend="rbln",
        # Disable dynamic shape support, as the RBLN backend currently does not support it
        dynamic=False,
        options={"cache_dir": f"./{model_name}"},
    )

    # (Optional) First call of forward invokes the compilation
    model(batch)

    # After that, You can use models compiled for RBLN hardware
    rbln_result = model(batch)

    # Display results
    class_idx = torch.argmax(rbln_result).item()
    print("Top1 Classification Index: ", class_idx)  # 281: "tabby, tabby cat"


if __name__ == "__main__":
    main()
