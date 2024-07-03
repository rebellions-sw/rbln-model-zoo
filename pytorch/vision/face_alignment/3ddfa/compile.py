import os
import sys
import yaml

# Pre-build scripts
sys.path.append(os.path.join(sys.path[0], "3DDFA_V2"))
os.system(f"cd {sys.path[0]}/3DDFA_V2/; ./build.sh; cd ../")

import torch

import rebel  # RBLN Compiler


from TDDFA import TDDFA


def main():
    # Load pre-defined Configuration
    cfg = yaml.load(open(sys.path[0] + "/3DDFA_V2/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    cfg["bfm_fp"] = sys.path[0] + "/3DDFA_V2/configs/bfm_noneck_v3.pkl"
    cfg["checkpoint_fp"] = sys.path[0] + "/3DDFA_V2/weights/mb1_120x120.pth"

    # Set model Envirnoment Variables
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "4"

    # Import models
    tddfa = TDDFA(**cfg)
    model = tddfa.model

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, 3, cfg["size"], cfg["size"]], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)
    compiled_model.save("./3ddfa.rbln")


if __name__ == "__main__":
    main()
