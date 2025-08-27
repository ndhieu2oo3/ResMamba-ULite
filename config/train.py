from config.segmentor import Segmentor
import pytorch_lightning as pl
import os
from module.model.remambaulite import ResMambaULite
from datasets.datasets import train_dataset, test_dataset

model = ResMambaULite().cuda()

os.makedirs('./weight/ISIC2018/', exist_ok = True)
check_point = pl.callbacks.model_checkpoint.ModelCheckpoint('./weight/ISIC2018/', filename="ckpt{val_dice:0.4f}",
                                                            monitor="val_dice", mode = "max", save_top_k =1,
                                                            verbose=True, save_weights_only=True,
                                                            auto_insert_metric_name=False)
progress_bar = pl.callbacks.TQDMProgressBar()
PARAMS = {"benchmark": True, "enable_progress_bar" : True,"logger":True,
          "callbacks" : [check_point, progress_bar],
          "log_every_n_steps" :1, "num_sanity_val_steps":0, "max_epochs":200,
          "precision":16,
          }
trainer = pl.Trainer(**PARAMS)
segmentor = Segmentor(model=model)
trainer.fit(segmentor, train_dataset, test_dataset)