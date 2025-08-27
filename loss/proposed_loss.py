import torch
from loss.loss import DiceLoss
import torch.nn.functional as F

def dilatted(mask, kernel_size=3):
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device).float()
    mask = mask.float()
    mask = F.conv2d(mask, kernel, padding="same")
    mask = torch.clip(mask, 0, 1)
    return mask

def erosin(mask, kernel_size=3):
    mask = mask.float()
    kernel = (torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)*(1/(kernel_size**2))).type_as(mask)

    mask = F.conv2d(mask, kernel, padding="same")
    mask = torch.floor(mask + 1e-2)
    return mask

def margin(mask, kernel_size):
    mask_dilated = dilatted(mask, kernel_size)
    mask_erosin = erosin(mask, kernel_size)
    return mask_dilated - mask_erosin

def marginweight(mask, weight_in=3, weight_out=5, weight_margin=2, kernel_size=7):
    mask_dilated = dilatted(mask, kernel_size)
    mask_erosin = erosin(mask, kernel_size)
    return (mask_dilated - mask_erosin)*weight_margin + mask_erosin*weight_in + (1 - mask_dilated)*weight_out

def MWLLoss(y_true, y_pred):
    y_pred = torch.sigmoid(y_pred)
    y_true = y_true.float()
    weight = marginweight(y_true, weight_in=3, weight_out=1, weight_margin=6, kernel_size=9)
    loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight = weight)
    return loss

def MarginalDice(y_true, y_pred):
    y_predict = torch.sigmoid(y_pred)
    y_true = y_true
    weight = marginweight(y_true)
    loss = weight*DiceLoss()(y_pred, y_true)
    return loss.mean()

def MWLTSPD (y_true, y_pred, m =2):
    y_pred = torch.sigmoid(y_pred)
    y_true = y_true
    loss = (y_true)*(1-y_pred**m) + (1-y_true)*(y_pred**m)
    loss = loss*marginweight(y_true)
    return loss.mean()

def MWL_add_dice (y_true, y_pred):
    y_pred = torch.sigmoid(y_pred)
    y_true = y_true
    weight = marginweight(y_true)
    return DiceLoss()(y_pred, y_true)*0.5 + MWLLoss(y_true, y_pred)*0.5

