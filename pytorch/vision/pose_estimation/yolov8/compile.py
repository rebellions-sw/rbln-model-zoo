import argparse

import rebel
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect, Pose


class RBLNYOLOv8PoseWrapper(nn.Module):
    def __init__(self, model):
        super(RBLNYOLOv8PoseWrapper, self).__init__()
        self.model = model.model
        self.save = model.save

    def forward(self, x):
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            if isinstance(m, Pose):
                """Perform forward pass through YOLO model and return predictions."""
                bs = x[0].shape[0]  # batch size
                kpt = torch.cat(
                    [m.cv4[i](x[i]).view(bs, m.nk, -1) for i in range(m.nl)], -1
                )  # (bs, 17*3, h*w)
                x = Detect.forward(m, x)
                pred_kpt = self.kpts_decode(m, bs, kpt)
                x = (
                    torch.cat([x, pred_kpt], 1)
                    if m.export
                    else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))
                )
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    @staticmethod
    def kpts_decode(m: Pose, bs, kpts):
        """Decodes keypoints."""
        ndim = m.kpt_shape[1]
        if m.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *m.kpt_shape, -1)
            new_y = torch.split(y, [2, y.shape[-1] - 2], dim=2)
            a = (new_y[0] * 2.0 + (m.anchors - 0.5)) * m.strides
            if ndim == 3:
                a = torch.cat((a, new_y[1].sigmoid()), 2)
            return a.view(bs, m.nk, -1)
        else:
            y = kpts
            if ndim == 3:
                new_y = torch.sigmoid(y[:, 2::3])
                y = y.slice_scatter(new_y, dim=1, start=2, step=3)
            new_y = (y[:, 0::ndim] * 2.0 + (m.anchors[0] - 0.5)) * m.strides
            y = y.slice_scatter(new_y, dim=1, start=0, step=ndim)
            new_y = (y[:, 1::ndim] * 2.0 + (m.anchors[1] - 0.5)) * m.strides
            y = y.slice_scatter(new_y, dim=1, start=1, step=ndim)
            return y


def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="yolov8s-pose",
        choices=["yolov8s-pose", "yolov8n-pose", "yolov8m-pose", "yolov8l-pose", "yolov8x-pose"],
        help="available model variations",
    )
    return parser.parse_args()


def main():
    args = parsing_argument()
    model_name = args.model_name

    model = YOLO(model_name + ".pt")
    model = model.model.eval()

    # Compile torch model for ATOM
    input_info = [
        ("input_np", [1, 3, 640, 640], torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(RBLNYOLOv8PoseWrapper(model), input_info)

    # Save compiled results to disk
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
