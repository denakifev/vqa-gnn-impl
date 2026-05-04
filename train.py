import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import (
    patch_hydra_argparse_compat,
    set_random_seed,
    setup_saving_and_logging,
)
from src.utils.model_freeze import apply_freeze_policy, count_parameters
from src.utils.optim import (
    normalize_optimizer_param_groups,
    resolve_effective_epoch_len,
    resolve_lr_scheduler_kwargs,
)

warnings.filterwarnings("ignore", category=UserWarning)
patch_hydra_argparse_compat()


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    apply_freeze_policy(model, config.get("freeze_policy"))
    logger.info(model)
    param_stats = count_parameters(model)
    logger.info(
        "Parameter accounting after freeze policy: "
        f"total={param_stats['total']:,}, "
        f"trainable={param_stats['trainable']:,}, "
        f"frozen={param_stats['frozen']:,}"
    )

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    if config.get("optimizer_param_groups") is not None:
        trainable_params = instantiate(config.optimizer_param_groups, model=model)
        trainable_params = normalize_optimizer_param_groups(trainable_params)
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params, _convert_="all")
    effective_epoch_len = resolve_effective_epoch_len(
        config.trainer.get("epoch_len"),
        dataloaders["train"],
    )
    lr_scheduler_kwargs = resolve_lr_scheduler_kwargs(
        config.lr_scheduler,
        optimizer,
        effective_epoch_len,
    )
    lr_scheduler = instantiate(config.lr_scheduler, _convert_="all", **lr_scheduler_kwargs)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
