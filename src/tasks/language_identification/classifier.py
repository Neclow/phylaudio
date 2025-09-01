# pylint: disable=invalid-name

"""Downstream LID classification"""

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import autograd, nn, optim
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from ..._config import NONE_TENSOR


# Warnings: https://github.com/pytorch/pytorch/issues/119475
class STEFunction(autograd.Function):
    """Straight-through estimator function"""

    @staticmethod
    def forward(ctx, i):
        return i.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class STE(nn.Module):
    """Straight-through estimator activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (*), where * means any number of dimensions.

        Returns
        -------
        output : torch.Tensor
            (*), where * means any number of dimensions.
        """
        return STEFunction.apply(x)


class MLP(nn.Module):
    """(Multi-layer) linear classifier

    Parameters
    ----------
    in_dim : int
        Input dimension
    out_dim : int
        Output dimension
    hidden_dim : int, optional
        Hidden dimension (for multiple layers), by default None
    ste : bool, optional
        Whether to use the STE activation function, by default False
    dtype : torch.dtype, optional
        torch dtype, by default None
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None, ste=False, dtype=None):
        super().__init__()

        # TODO: add num_layers as arg for classifier?

        self.in_dim = in_dim

        self.out_dim = out_dim

        self.hidden_dim = hidden_dim

        if self.hidden_dim is None:
            self.projector = nn.Identity()

            self.classifier = nn.Linear(self.in_dim, self.out_dim, dtype=dtype)
        else:
            self.activation = STE() if ste else nn.ReLU()

            self.projector = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim, dtype=dtype),
                self.activation,
            )

            self.classifier = nn.Linear(self.hidden_dim, self.out_dim, dtype=dtype)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            (B, *, D), where * means any number of dimensions.
            B = batch size
            D = input dimension

            3D: N = number of chunks

        Returns
        -------
        output : torch.Tensor
            (B, C)

            C = number of classes
        """
        # Feature extraction: B x T* --> B x (*) x D

        # Projection: B x (*) x D -> B x (*) x D'
        x = self.projector(x)

        # 2D: B x D' --> B X C
        # 3D: B x N x D' --> B x N x C
        x = self.classifier(x)

        # 2D: B x C --> B x C (logits)
        # 3D: B x N x C --> B x N x C (logits)
        x = self.log_softmax(x)

        if x.ndim == 3:
            # B x N x C --> B x C
            x = x.mean(1)

        return x


class LightningMLP(LightningModule):
    def __init__(
        self, feature_extractor, num_classes, loss_fn, lr, weight_decay, hidden_dim=None
    ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()

        self.classifier = MLP(
            in_dim=self.feature_extractor.emb_dim,
            out_dim=num_classes,
            dtype=self.feature_extractor.dtype,
            hidden_dim=hidden_dim,
        )
        self.classifier.train()

        self.loss_fn = loss_fn

        metric = MetricCollection(
            [
                MulticlassAccuracy(num_classes=num_classes, average="micro"),
                MulticlassF1Score(num_classes=num_classes, average="macro"),
            ]
        )

        self.train_metric = metric

        self.valid_metric = metric.clone()

        self.test_metric = metric.clone()

        self.lr = lr

        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["feature_extractor"])

    def training_step(self, *args, **kwargs):
        batch = args[0]

        loss, score = self._get_loss_and_scores(batch, self.train_metric)

        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": score["MulticlassAccuracy"],
                "train_f1": score["MulticlassF1Score"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, *args, **kwargs):
        batch = args[0]

        loss, score = self._get_loss_and_scores(batch, self.valid_metric)

        self.log_dict(
            {
                "valid_loss": loss,
                "valid_accuracy": score["MulticlassAccuracy"],
                "valid_f1": score["MulticlassF1Score"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, *args, **kwargs):
        batch = args[0]

        _, score = self._get_loss_and_scores(batch, self.test_metric)

        self.log_dict(
            {
                "test_accuracy": score["MulticlassAccuracy"],
                "test_f1": score["MulticlassF1Score"],
            },
            on_step=False,
            on_epoch=True,
        )

    def _get_loss_and_scores(self, batch, metric):
        self.feature_extractor.eval()

        # For multimodel
        if isinstance(batch, list):
            X_batch = [x["input"] for x in batch]
            y_batch = batch[0]["label"]  # Expecting same labels across stacked datasets
            a_batch = [x["attention_mask"] for x in batch]
        else:
            # Forward pass
            X_batch, y_batch, a_batch = (
                batch["input"],
                batch["label"],
                batch["attention_mask"],
            )

        with torch.no_grad():
            if torch.equal(a_batch[0], NONE_TENSOR.to(a_batch[0].device)):
                emb_batch = self.feature_extractor(X_batch)
            else:
                emb_batch = self.feature_extractor(X_batch, a_batch)

        o_batch = self.classifier(emb_batch)

        # Compute losses
        loss = self.loss_fn(o_batch, y_batch)

        # Update metrics
        score = metric(o_batch, y_batch)

        return loss, score

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return optimizer
