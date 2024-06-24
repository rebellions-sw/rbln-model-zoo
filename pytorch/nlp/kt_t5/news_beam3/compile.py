import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.rbln_t5_utils import t5_compile_opt


def parsing_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weight_path", type=str, help="(str) path of weight file", required=True)
    parser.add_argument(
        "--rbln_model_path",
        type=str,
        default="./",
        help="(str) base directory to save compiled model",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()

    model_path = Path(args.rbln_model_path)
    if not os.path.isdir(model_path):
        model_path.mkdir(parents=True)

    batch_size = 1
    num_beams = 3

    # build 3 models with different seq combination
    # one model will be selected based on encoder input sequence
    input_seq_list = [384, 448, 512]
    max_output_seq_length = 256

    # Compile
    for input_seq_length in input_seq_list:
        compiled_model = t5_compile_opt(
            input_seq_length, max_output_seq_length, args.weight_path, batch_size, num_beams
        )

        # Save compiled results to disk
        save_path = os.path.join(
            args.rbln_model_path,
            f"i_seq_{input_seq_length}_o_seq_{max_output_seq_length}_opt.rbln",
        )
        compiled_model.save(save_path)


if __name__ == "__main__":
    main()
