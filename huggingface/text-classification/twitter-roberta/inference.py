import os
import argparse
import csv
import urllib.request
import torch
import numpy as np
from transformers import RobertaTokenizerFast
from optimum.rbln import RBLNRobertaForSequenceClassification


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        default="Celebrating my promotion 😎",
        help="(str) type, text for score",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="emotion",
        help="(str) model task,",
    )
    return parser.parse_args()


# Preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def download_label_mapping(task):
    mapping_link = (
        f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    )
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode("utf-8").split("\n")
        csvreader = csv.reader(html, delimiter="\t")
    return [row[1] for row in csvreader if len(row) > 1]


def main():
    args = parsing_argument()
    model_id = f"cardiffnlp/twitter-roberta-base-{args.task}"

    # Load compiled model
    model = RBLNRobertaForSequenceClassification.from_pretrained(
        model_id=os.path.basename(model_id),
        export=False,
    )

    # Prepare inputs
    tokenizer = RobertaTokenizerFast.from_pretrained(os.path.basename(model_id))
    labels = download_label_mapping(args.task)

    # Encode the text
    text = preprocess(args.text)
    inputs = tokenizer.batch_encode_plus(
        [text], max_length=512, truncation=True, padding="max_length", return_tensors="pt"
    )

    # Run the model
    output = model(inputs.input_ids, inputs.attention_mask)

    # Show text and result
    print("--- text ---")
    print(args.text)
    print("--- score ---")

    num_class = 4
    for batch_itr in range(output.shape[0]):
        # Apply softmax to get probabilities
        scores = output[batch_itr].detach()
        scores = torch.nn.functional.softmax(scores, dim=-1).numpy()

        # Get ranking of scores
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        # Print out the results
        for i in range(scores.shape[0]):
            label = labels[ranking[i]]
            score = scores[ranking[i]]
            print(f"{batch_itr}) {label} {np.round(float(score), num_class)}")


if __name__ == "__main__":
    main()
