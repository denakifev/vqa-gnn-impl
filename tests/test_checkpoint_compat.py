from pathlib import Path

import torch
from omegaconf import OmegaConf


def test_checkpoint_config_is_saved_as_plain_python_container(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "model": {"name": "gqa"},
            "optimizer": {"lr": 1e-4},
        }
    )
    state = {
        "epoch": 1,
        "state_dict": {"weight": torch.ones(1)},
        "config": OmegaConf.to_container(cfg, resolve=False),
    }

    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(state, checkpoint_path)

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert isinstance(loaded["config"], dict)
    assert loaded["config"]["optimizer"]["lr"] == 1e-4
