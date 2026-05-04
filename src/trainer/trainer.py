from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if self.writer is None:
            return

        if "graph_link_stats" in batch:
            for stat_name, stat_value in batch["graph_link_stats"].items():
                self.writer.add_scalar(f"graph_link/{stat_name}", float(stat_value))

        baseline_logits = batch.get("baseline_logits")
        if baseline_logits is not None:
            baseline_logit_norm = torch.linalg.vector_norm(
                baseline_logits.detach(), dim=-1
            ).mean()
            self.writer.add_scalar(
                "graph_link/baseline_logit_norm", float(baseline_logit_norm.cpu().item())
            )

        graph_link_logits = batch.get("graph_link_logits")
        if graph_link_logits is not None:
            graph_link_logit_norm = torch.linalg.vector_norm(
                graph_link_logits.detach(), dim=-1
            ).mean()
            self.writer.add_scalar(
                "graph_link/logit_norm", float(graph_link_logit_norm.cpu().item())
            )
            self.writer.add_scalar(
                "graph_link/logit_abs_mean",
                float(graph_link_logits.detach().abs().mean().cpu().item()),
            )

            link_alpha = batch.get("graph_link_stats", {}).get("link_alpha")
            if link_alpha is not None:
                scaled_link_logit_norm = torch.linalg.vector_norm(
                    (float(link_alpha) * graph_link_logits.detach()),
                    dim=-1,
                ).mean()
                self.writer.add_scalar(
                    "graph_link/scaled_logit_norm",
                    float(scaled_link_logit_norm.cpu().item()),
                )
