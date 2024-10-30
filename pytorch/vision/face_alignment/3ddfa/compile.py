import os
import sys
import yaml
import torch
import rebel
import re


def update_file(file_path, replacements):
    with open(file_path, "r") as f:
        content = f.read()
    for pattern, replacement in replacements:
        content = re.sub(r"\b" + re.escape(pattern) + r"\b", replacement, content)
    with open(file_path, "w") as f:
        f.write(content)


def update_submodule():
    base_path = f"{sys.path[0]}/3DDFA_V2"
    os.system(f"cd {base_path} && git reset --hard && cd ..")
    print("Updating submodule for NumPy and dtype compatibility...")

    files_to_update = {
        "FaceBoxes/utils/nms/cpu_nms.pyx": [
            ("np.int_t", "np.int64_t"),
            ("np.float_t", "np.float32_t"),
            ("long", "int"),
            ("np.int32_t", "np.int64_t"),
            ("np.int", "np.int64"),
        ],
        "bfm/bfm.py": [
            ("np.long", "np.int64"),
        ],
        "FaceBoxes/FaceBoxes_ONNX.py": [
            ("np.int", "np.int64"),
            ("np.float", "np.float32"),
        ],
    }

    for file, replacements in files_to_update.items():
        update_file(os.path.join(base_path, file), replacements)

    print("Files updated successfully.")


def main():
    # Update submodule to support NumPy >= 1.24.0 by replacing deprecated types with supported ones
    update_submodule()

    # Add 3DDFA_V2 to Python path
    sys.path.append(os.path.join(sys.path[0], "3DDFA_V2"))

    # Run build script
    os.system(f"cd {sys.path[0]}/3DDFA_V2/; ./build.sh; cd ../")

    # Now import TDDFA after adding to path and building
    from TDDFA import TDDFA

    cfg = yaml.safe_load(open(f"{sys.path[0]}/3DDFA_V2/configs/mb1_120x120.yml"))
    cfg.update(
        {
            "bfm_fp": f"{sys.path[0]}/3DDFA_V2/configs/bfm_noneck_v3.pkl",
            "checkpoint_fp": f"{sys.path[0]}/3DDFA_V2/weights/mb1_120x120.pth",
        }
    )

    os.environ.update({"KMP_DUPLICATE_LIB_OK": "True", "OMP_NUM_THREADS": "4"})

    tddfa = TDDFA(**cfg)
    input_info = [("input_np", [1, 3, cfg["size"], cfg["size"]], torch.float32)]
    compiled_model = rebel.compile_from_torch(tddfa.model, input_info)
    compiled_model.save("./3ddfa.rbln")


if __name__ == "__main__":
    main()
