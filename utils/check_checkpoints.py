"""Load every PatchTST checkpoint in the repo and run one dummy batch through it.

    python -m utils.check_checkpoints
"""
import os

import torch

from baseline.patchtst.patchtst_model import PatchTSTModel

SENSORS = ["left_arm", "left_leg", "right_arm", "right_leg"]
CONFIGS = {
    "checkpoints_patchtst": dict(d_model=64, num_layers=3, n_heads=4),
    "checkpoints_patchtst_aug": dict(d_model=128, num_layers=6, n_heads=8),
}

if __name__ == "__main__":
    for folder, cfg in CONFIGS.items():
        for sensor in SENSORS:
            path = os.path.join(folder, f"patchtst_{sensor}.pth")
            model = PatchTSTModel(input_channels=3, patch_size=10, seq_len=50,
                                  num_classes=19, dropout=0.1, **cfg)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(2, 50, 3))
            n_params = sum(p.numel() for p in model.parameters())
            print(f"OK  {path}  params={n_params:,}  output={tuple(out.shape)}")
