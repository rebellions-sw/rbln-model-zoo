import argparse
import os
import urllib
from transformers import AutoFeatureExtractor
from optimum.rbln import RBLNResNetForImageClassification
from PIL import Image


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet-50",
        help="(str) model type, image classification model name.",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_id = f"microsoft/{args.model_name}"

    # Load compiled model
    model = RBLNResNetForImageClassification.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    img_url = "https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/tabby.jpg"
    img_path = "./tabby.jpg"
    if not os.path.exists(img_path):
        with urllib.request.urlopen(img_url) as response, open(img_path, "wb") as f:
            f.write(response.read())

    image = Image.open(img_path)

    image_processor = AutoFeatureExtractor.from_pretrained(model_id)
    inputs = image_processor([image], return_tensors="pt")

    # Run inference
    logits = model(**inputs)
    labels = logits.argmax(-1)

    # Show results
    print("predicted label:", [model.config.id2label[label.item()] for label in labels])


if __name__ == "__main__":
    main()
