import pytorch_lightning as L
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from .config import TrainerConfig
from ...experiment.config import ExperimentConfig, ExpType
from .config import Losses
from ..dataset.dataset import Dataset
from ..autoencoder.model import Model as Autoencoder
from ..unet.model import Model as Unet
from typing import Tuple, Callable, Dict, Optional, Type
import time
import math


_Prediction = Tensor
_Loss = Tensor
_Input = Tensor
_Target = Tensor

_loss_dict: Dict[Losses, Callable[[], nn.Module]] = {
    Losses.L1: torch.nn.L1Loss,
    Losses.MSE: torch.nn.MSELoss,
    Losses.CUSTOM_L1: torch.nn.L1Loss,
    Losses.CUSTOM_MSE: torch.nn.MSELoss,
}

_exp_type_to_model_class: Dict[ExpType, Type[nn.Module]] = {
    ExpType.AUTOENCODER: Autoencoder,
    ExpType.UNET: Unet
}


class _LightningWrapper(L.LightningModule):
    """This wraps the torch module to enable usage with lightning trainers"""

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self._cfg = cfg
        self._model = _exp_type_to_model_class[cfg.exp_type](cfg.model)
        self._loss_fn = self._build_loss_fn()
        self._val_sample_freq = None
        self._train_sample_freq = None

    def set_val_sample_freq(self, val_dataloader: DataLoader) -> None:
        self._val_sample_freq = math.ceil(
            len(val_dataloader) / self._cfg.trainer.viz_sample_frequency
        )

    def set_train_sample_freq(self, train_dataloader: DataLoader) -> None:
        self._train_sample_freq = math.ceil(
            len(train_dataloader) / self._cfg.trainer.viz_sample_frequency
        )

    def _build_loss_fn(self) -> Callable[[_Input, _Prediction, _Target], _Loss]:
        base_fn = _loss_dict[self._cfg.trainer.loss]()
        if self._cfg.trainer.loss is Losses.CUSTOM_MSE:
            def loss_function(input_: _Input, prediction: _Prediction, target: _Target) -> _Loss:
                # Histogram loss
                boundaries = torch.arange(0, 1.00001, 1.0 /20, device=prediction.device)
                bucketized = torch.bucketize(target, boundaries=boundaries)
                _, inverses, counts = torch.unique(bucketized, return_counts=True, return_inverse=True)
                f = counts[inverses]
                return torch.sum(torch.sqrt(torch.square(prediction - target)) / f)
        elif self._cfg.trainer.loss is Losses.CUSTOM_L1:
            def loss_function(input_: _Input, prediction: _Prediction, target: _Target) -> _Loss:
                # Histogram loss
                boundaries = torch.arange(0, 1.00001, 1.0 /20, device=prediction.device)
                bucketized = torch.bucketize(target, boundaries=boundaries)
                _, inverses, counts = torch.unique(bucketized, return_counts=True, return_inverse=True)
                f = counts[inverses]
                return torch.sum(torch.abs(prediction - target) / f)
        else:
            def loss_function(
                input_: _Input, prediction: _Prediction, target: _Target
            ) -> _Loss:
                return base_fn(prediction, target)
        return loss_function

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self._model.parameters(), lr=self._cfg.trainer.lr)

    def _common_step(
        self, input_batch: Tuple[Tensor, Tensor]
    ) -> Tuple[_Prediction, _Loss]:
        """Common step between train and validation step. Do forward pass and return prediction and loss"""
        x, y = input_batch
        y_hat = self(x)
        loss = self._loss_fn(x, y_hat, y)
        return y_hat, loss

    def training_step(self, train_batch: Tuple[Tensor, Tensor], batch_idx) -> Tensor:
        x, y = train_batch
        y_hat, loss = self._common_step(train_batch)
        self.log_dict(
            {
                "train_loss": loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if batch_idx % self._train_sample_freq == 0:
            self.logger.experiment.add_image(
                "training_sample", self.sample_img(x, y_hat, y), self.global_step
            )
        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        y_hat, loss = self._common_step(val_batch)
        self.log_dict(
            {
                "val_loss": loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if batch_idx % self._val_sample_freq == 0:
            self.logger.experiment.add_image(
                "val_sample", self.sample_img(x, y_hat, y), self.global_step + batch_idx
            )

    def sample_img(self, x, y_hat, y):
        batch_size = x.shape[0]
        inv = math.ceil(batch_size / self._cfg.trainer.vis_sample_num)

        c1 = x[::inv, None, 0]
        c2 = x[::inv, None, 1]
        c3 = y_hat[::inv]
        c4 = y[::inv]
        grid = torch.concat((c1, c2, c3, c4), 1).view(-1, 1, c3.shape[-2], c3.shape[-1])
        grid = torchvision.utils.make_grid(grid, nrow=4, normalize=True)
        return grid


class Trainer:
    def __init__(self, cfg: ExperimentConfig):
        self._exp_cfg = cfg
        self._cfg: TrainerConfig = cfg.trainer
        self._model = _LightningWrapper(cfg)
        self._train_dataset = Dataset(cfg.dataset_train)
        self._val_dataset: Optional[Dataset] = None
        if cfg.trainer.use_validation:
            self._val_dataset = Dataset(cfg.dataset_val)

    def _get_max_batch_size_train(self) -> int:
        # Save model and optimizer states to reset after calculation
        model_state = self._model._model.state_dict()
        model = self._model._model
        model.cuda()
        loss_fn = self._model._loss_fn
        optimizer = self._model.configure_optimizers()
        optimizer_state = optimizer.state_dict()
        # Run training loops to calculate the bets batch size
        batch_size = 0
        best_dps = 0
        best_batch_size = 1
        input_size = self._exp_cfg.dataset_train.input_size
        while True:
            batch_size += 1
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                x = torch.rand((batch_size, 2, *input_size), device="cpu")
                y = torch.rand((batch_size, 1, *input_size), device="cpu")
                # Run one batch to run memory allocation first
                x_ = x.cuda()
                y_ = y.cuda()
                optimizer.zero_grad()
                y_hat = model(x_)
                loss = loss_fn(x_, y_, y_hat)
                loss.backward()
                optimizer.step()
                # Start timer
                start = time.time()
                for _ignored in range(20):
                    x_ = x.cuda()
                    y_ = y.cuda()
                    optimizer.zero_grad()
                    y_hat = model(x_)
                    loss = loss_fn(x_, y_, y_hat)
                    loss.backward()
                    optimizer.step()
                duration = time.time() - start
                memory_reserved = torch.cuda.memory_reserved(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if memory_reserved > total_memory:
                    break
                dps = batch_size * 20 / duration
                if dps > best_dps:
                    best_batch_size = batch_size
            except RuntimeError:
                break
        # Reset optimizer and model states and clear cache
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        del x_
        del y_
        del x
        del y
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Return best batch size
        return best_batch_size

    def _get_max_batch_size_val(self) -> int:
        # Save model and optimizer states to reset after calculation
        model = self._model._model
        model.cuda()
        loss_fn = self._model._loss_fn
        # Run validation loops to calculate the bets batch size
        batch_size = 0
        best_dps = 0
        best_batch_size = 1
        input_size = self._exp_cfg.dataset_train.input_size
        while True:
            batch_size += 1
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                x = torch.rand(
                    (batch_size, 2, *input_size), device="cpu", requires_grad=False
                )
                y = torch.rand(
                    (batch_size, 1, *input_size), device="cpu", requires_grad=False
                )
                # Run one batch to run memory allocation first
                x_ = x.cuda()
                y_ = y.cuda()
                with torch.no_grad():
                    y_hat = model(x_)
                    loss_fn(x_, y_, y_hat)
                # Start timer
                with torch.no_grad():
                    start = time.time()
                    for _ignored in range(20):
                        x_ = x.cuda()
                        y_ = y.cuda()
                        y_hat = model(x_)
                        loss_fn(x_, y_, y_hat)
                    duration = time.time() - start
                dps = batch_size * 20 / duration
                memory_reserved = torch.cuda.memory_reserved(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if memory_reserved > total_memory:
                    break
                if dps > best_dps:
                    best_batch_size = batch_size
            except RuntimeError as e:
                break
        del x_
        del y_
        del x
        del y
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return best_batch_size

    def _get_data_loaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        # First, calculate batch size
        batch_size_train = self._cfg.batch_size_train
        if batch_size_train is None:
            print("\rFinding best batch size for training...", end="")
            batch_size_train = self._get_max_batch_size_train()
            print("\rBest batch size train:", batch_size_train, "          ")
        train_loader = DataLoader(self._train_dataset, batch_size_train, shuffle=True)
        val_loader = None
        if self._cfg.use_validation:
            batch_size_val = self._cfg.batch_size_val
            if batch_size_val is None:
                print("\rFinding best batch size for validation... ", end="")
                batch_size_val = self._get_max_batch_size_val()
                print("\rBest batch size val:", batch_size_val)

            val_loader = DataLoader(self._val_dataset, batch_size_val, shuffle=False)
        return train_loader, val_loader

    def train(self, caller):
        if caller != "__main__":
            return
        exp_type = self._exp_cfg.exp_type
        output_path = self._cfg.out_path
        logger = TensorBoardLogger(f"{output_path}/{exp_type.value.lower()}")
        logger.log_hyperparams(self._exp_cfg.to_dict())
        monitoring_value = "val_loss" if self._cfg.use_validation else "train_loss"
        callbacks = [
            ModelCheckpoint(
                filename="best_loss",
                monitor=monitoring_value,
                mode="min",
                save_last=True,
            )
        ]
        if self._cfg.early_stopping_patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitoring_value, patience=self._cfg.early_stopping_patience
                )
            )
        train_loader, val_loader = self._get_data_loaders()
        self._model.set_train_sample_freq(train_loader)
        if self._cfg.use_validation:
            self._model.set_val_sample_freq(val_loader)
        trainer_params = dict(
            enable_progress_bar=self._cfg.progress_bar,
            logger=logger,
            accelerator=self._exp_cfg.accelerator,
            devices=self._cfg.devices,
            max_epochs=self._cfg.epochs,
            log_every_n_steps=1,
            callbacks=callbacks,
        )
        if self._cfg.devices != 1:
            trainer_params["strategy"] = "ddp"
        trainer = L.Trainer(**trainer_params)
        if val_loader is None:
            trainer.fit(model=self._model, train_dataloaders=train_loader, ckpt_path=self._cfg.resume)
        else:
            trainer.fit(
                model=self._model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=self._cfg.resume
            )
