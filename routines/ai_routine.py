import pytorch_lightning as pl
import torch

from terrace.comp_node import Input
from models.make_model import make_model
from datasets.make_dataset import make_dataloader
from common.losses import get_losses
from common.metrics import get_metrics

class AIRoutine(pl.LightningModule):
    """ A "Routine" is basically just a lightning module -- something that's model
    agnostic, but defines how the training and eval is performed for a particular
    type of model (e.g. activitity prediction) """

    @classmethod
    def from_checkpoint(cls, cfg, checkpoint_file):
        return cls.load_from_checkpoint(checkpoint_file, cfg=cfg, checkpoint_file=checkpoint_file)

    def __init__(self, cfg, checkpoint_file = None):
        super().__init__()
        self.checkpoint_file = checkpoint_file
        self.cfg = cfg
        self.learn_rate = cfg.learn_rate
        self.train_dataloader = make_dataloader(cfg, "train")
        self.val_dataloader = make_dataloader(cfg, "val")
        self.val_variance = self.val_dataloader.dataset.get_variance()
        in_node = Input(self.val_dataloader.get_type_data())
        self.model = make_model(cfg, in_node)

    def forward(self, batch):
        act = self.model(batch)
        return act

    def shared_eval(self, batch, batch_idx, prefix):
        pred = self(batch)
        loss, loss_dict = get_losses(self.cfg, batch, pred)
        self.log(f"{prefix}_loss", loss, prog_bar=True, batch_size=len(batch))
        for key, val in loss_dict.items():
            self.log(f"{prefix}_{key}", val, prog_bar=False, batch_size=len(batch))
        metrics = get_metrics(self.cfg, batch, pred, self.val_variance)
        for key, val in metrics.items():
            self.log(f"{prefix}_{key}", val, prog_bar=False, batch_size=len(batch))
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learn_rate)

    def fit(self, logger, callbacks):
        gpus = int(torch.cuda.is_available())
        trainer = pl.Trainer(gpus=gpus,
                             max_epochs=self.cfg.max_epochs,
                             val_check_interval=self.cfg.val_check_interval,
                             logger=logger,
                             callbacks=callbacks,
                             resume_from_checkpoint=self.checkpoint_file)
        trainer.fit(self, self.train_dataloader, self.val_dataloader)
        # trainer.validate(self, self.val_dataloader)