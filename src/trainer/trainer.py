from random import randint

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


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

        outputs = self.model(**batch)
        batch.update(outputs)

        batch["generated_wav"] = batch["generated_wav"].detach()
        outputs = self.model.wav_generator.discriminate(**batch, detach=True)
        batch.update(outputs)

        if self.is_train:
            self.d_optimizer.zero_grad()
            d_loss = self.d_criterion(**batch)
            batch.update(d_loss)
            batch["loss_disc"].backward()
            self._clip_grad_norm()
            self.d_optimizer.step()

            self.g_optimizer.zero_grad()
            outputs = self.model.wav_generator.discriminate(**batch)
            batch.update(outputs)

            g_loss = self.g_criterion(**batch)
            batch.update(g_loss)
            batch["loss_gen"].backward()
            self._clip_grad_norm()
            self.g_optimizer.step()

            if self.d_lr_scheduler is not None:
                self.d_lr_scheduler.step()
            if self.g_lr_scheduler is not None:
                self.g_lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        if self.is_train:
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
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
        else:
            self.log_spectrogram(**batch)
            self.log_audio(**batch)

    def log_spectrogram(self, gt_melspectrogram, gen_melspectrogram, **batch):
        idx = randint(0, len(gt_melspectrogram) - 1)

        gt_spectrogram = gt_melspectrogram[idx].detach().cpu()
        gen_spectrogram = gen_melspectrogram[idx].detach().cpu()
        gt_image = plot_spectrogram(gt_spectrogram)
        gen_image = plot_spectrogram(gen_spectrogram)

        self.writer.add_image("gt_melspectrogram", gt_image)
        self.writer.add_image("gen_spectrogram", gen_image)

    def log_audio(self, audio, generated_wav, sample_rate, **batch):
        idx = randint(0, len(audio) - 1)

        gt_wav = audio[idx].detach().cpu()
        gen_wav = generated_wav[idx].detach().cpu()

        self.writer.add_audio("gt_wav", gt_wav, sample_rate=sample_rate)
        self.writer.add_audio("gen_wav", gen_wav, sample_rate=sample_rate)
