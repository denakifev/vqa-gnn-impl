import math

from omegaconf import DictConfig, ListConfig
from torch.optim.lr_scheduler import LambdaLR


def make_gqa_optimizer_param_groups(
    model,
    encoder_lr: float = 2e-5,
    decoder_lr: float = 2e-4,
    weight_decay: float = 1e-2,
):
    """
    Build paper-closer AdamW parameter groups for the active GQA path.

    The GQA paper uses a smaller LR for RoBERTa-L and a larger LR for the
    multimodal GNN/classifier stack. We also exclude biases and norm weights
    from weight decay, following the common AdamW setup for transformer models.
    """

    no_decay_terms = ("bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight")

    groups = {
        ("encoder", "decay"): {"params": [], "lr": encoder_lr, "weight_decay": weight_decay},
        ("encoder", "no_decay"): {"params": [], "lr": encoder_lr, "weight_decay": 0.0},
        ("decoder", "decay"): {"params": [], "lr": decoder_lr, "weight_decay": weight_decay},
        ("decoder", "no_decay"): {"params": [], "lr": decoder_lr, "weight_decay": 0.0},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        scope = "encoder" if name.startswith("question_encoder.encoder.") else "decoder"
        decay_key = "no_decay" if any(term in name for term in no_decay_terms) else "decay"
        groups[(scope, decay_key)]["params"].append(param)

    return [group for group in groups.values() if group["params"]]


def build_linear_warmup_cosine_scheduler(
    optimizer,
    n_epochs: int,
    epoch_len: int,
    warmup_ratio: float = 0.3,
    min_lr_scale: float = 0.0,
):
    """
    Per-step linear warmup followed by cosine decay.

    This mirrors the paper at a high level while staying practical for shorter
    Kaggle runs: the warmup duration is expressed as a ratio of the total run
    instead of a fixed epoch count.
    """

    total_steps = max(int(n_epochs) * int(epoch_len), 1)
    warmup_steps = int(round(total_steps * float(warmup_ratio)))
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))
    min_lr_scale = float(min_lr_scale)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        if total_steps <= warmup_steps:
            return 1.0

        progress = (current_step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def normalize_optimizer_param_groups(params):
    """
    Convert OmegaConf containers returned through Hydra into plain Python
    lists/dicts so torch.optim can consume param groups safely.
    """

    if isinstance(params, DictConfig):
        return {key: normalize_optimizer_param_groups(value) for key, value in params.items()}
    if isinstance(params, ListConfig):
        return [normalize_optimizer_param_groups(value) for value in params]
    return params
