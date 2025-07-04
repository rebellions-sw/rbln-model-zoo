import argparse
import os

from optimum.rbln import RBLNLlamaForCausalLM

DEFAULT_TP_SIZE = {
    "Midm-2.0-Base-Instruct": 8,
    "Midm-2.0-Mini-Instruct": 4,
}


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        choices=["Midm-2.0-Base-Instruct", "Midm-2.0-Mini-Instruct"],
        default="Midm-2.0-Mini-Instruct",
        help="(str) model type, Size of midm. [Midm-2.0-Base-Instruct, Midm-2.0-Mini-Instruct]",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()

    model_name = f"K-intelligence/{args.model_id}"

    model = RBLNLlamaForCausalLM.from_pretrained(
        model_id=os.path.basename(model_name),
        export=True,
        rbln_batch_size=1,
        rbln_max_seq_len=32_768,
        rbln_tensor_parallel_size=DEFAULT_TP_SIZE[os.path.basename(model_name)],
    )

    model.save_pretrained(os.path.basename(model_name))


if __name__ == "__main__":
    main()
