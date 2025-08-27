import pytorch_lightning as pl
import torch 
from loss.proposed_loss import MarginalDice
from metrics.metrics import dice_score, iou_score

class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch
        y_pred = self.model(image)
        loss = MarginalDice(y_true, y_pred)
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        return loss, dice, iou

    def training_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"loss": loss, "train_dice": dice, "train_iou": iou}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"val_loss":loss, "val_dice": dice, "val_iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, dice, iou = self._step(batch)
        metrics = {"loss":loss, "test_dice": dice, "test_iou": iou}
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                         factor = 0.5, patience=5, verbose =True)
        lr_schedulers = {"scheduler": scheduler, "monitor": "val_dice"}
        return [optimizer], lr_schedulers