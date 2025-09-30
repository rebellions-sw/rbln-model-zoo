import torch


def apply_sincos_to_pos_embed(pose_emb):
    cos_ = torch.cos(pose_emb)
    sin_ = torch.sin(pose_emb)
    return [cos_, sin_]


def run_model(model, *args, **kwargs):
    return model(*args, **kwargs)
