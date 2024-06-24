import os
import sys
import cv2
import yaml
import numpy as np

sys.path.append(os.path.join(sys.path[0], "3DDFA_V2"))
from utils.render_ctypes import render
from utils.functions import crop_img, parse_roi_box_from_bbox

from TDDFA import TDDFA

import onnxruntime
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX

import rebel  # RBLN Runtime


def main():
    # Load pre-defined Configuration
    cfg = yaml.load(open(sys.path[0] + "/3DDFA_V2/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
    cfg["bfm_fp"] = sys.path[0] + "/3DDFA_V2/configs/bfm_noneck_v3.pkl"
    cfg["checkpoint_fp"] = sys.path[0] + "/3DDFA_V2/weights/mb1_120x120.pth"

    # Set model Envirnoment Variables
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "4"

    # for configs
    tddfa = TDDFA(**cfg)

    # Load Face detection ONNX model
    face_detector = FaceBoxes_ONNX()
    face_detector.session = onnxruntime.InferenceSession(
        sys.path[0] + "/3DDFA_V2/FaceBoxes/weights/FaceBoxesProd.onnx", None
    )

    # Prepare input image
    image_name = "emma"
    img_file_path = sys.path[0] + f"/3DDFA_V2/examples/inputs/{image_name}.jpg"
    original_image = cv2.imread(img_file_path)
    boxes = face_detector(original_image)

    # Load 3DDFA model
    module = rebel.Runtime("./3ddfa.rbln")

    roi_box_list = []
    param_list = []

    # by multiple face box
    for box in boxes:
        roi_box = parse_roi_box_from_bbox(box)
        roi_box_list.append(roi_box)

        # preprocessing image
        image = crop_img(original_image, roi_box)
        image = cv2.resize(image, dsize=(cfg["size"], cfg["size"]), interpolation=cv2.INTER_LINEAR)
        image = tddfa.transform(image).unsqueeze(0)
        image = np.ascontiguousarray(image, dtype=np.float32)

        # run 3ddfa model
        param = module.run(image)

        # re-scale output
        param = param.flatten().astype(np.float32)
        param = param * tddfa.param_std + tddfa.param_mean
        param_list.append(param)

    # Extract Face Alignment Image
    ver_dense = tddfa.recon_vers(param_list, roi_box_list, dense_flag=True)
    img_draw = original_image.copy()
    img_draw = render(img_draw, ver_dense, tddfa.tri, wfp=f"./{image_name}_3ddfa.jpg", alpha=0.6)


if __name__ == "__main__":
    main()
