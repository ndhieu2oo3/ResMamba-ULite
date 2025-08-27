import pytorch_lightning as pl
from loss.proposed_loss import MarginalDice
from metrics.metrics import dice_score, iou_score
from module.model.remambaulite import ResMambaULite 
from datasets.datasets import test_dataset

class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        loss = MarginalDice(y_true, y_pred)
        print(loss.cpu().numpy(), end=' ')
        # Calculate Dice and IoU scores
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        # Compute TP, FP, FN, TN for Precision, Recall, and F-Score
        y_pred_bin = (y_pred > 0.5).float()  # Threshold predictions to binary
        TP = (y_pred_bin * y_true).sum(dim=(1, 2, 3))
        FP = ((y_pred_bin == 1) & (y_true == 0)).sum(dim=(1, 2, 3))
        FN = ((y_pred_bin == 0) & (y_true == 1)).sum(dim=(1, 2, 3))
        # Precision, Recall, and F-Score
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        # Average metrics over the batch
        precision_mean = precision.mean().item()
        recall_mean = recall.mean().item()
        f_score_mean = f_score.mean().item()
        # Log all metrics
        metrics = {
            "Test Dice": dice,
            "Test IoU": iou,
            "Test Precision": precision_mean,
            "Test Recall": recall_mean,
            "Test F-Score": f_score_mean,
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics
model = ResMambaULite().cuda()
model.eval()
# Dataset & Data Loader
CHECKPOINT_PATH = "./weight/isic2018/.ckpt"
# Prediction
trainer = pl.Trainer()
segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model = model)
trainer.test(segmentor, test_dataset)