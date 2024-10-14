from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from cosine import WarmupCosineScheduler
from Phase2_asr.transforms import TextTransform
import logging

# for testing
from espnet.asr.asr_utils import add_results_to_json
from pytorch_lightning import LightningModule
from Phase2_asr.asr_model import ASR




class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)  #这个是为了保存config文件到模型里，加载模型的时候就可以看到。
        self.cfg = cfg

        self.backbone_args = self.cfg.model.visual_backbone

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = ASR(len(self.token_list), cfg)

        # if use pretrained model to finetune
        if self.cfg.pretrained_model:
            ckpt = torch.load(
                self.cfg.pretrained_model, map_location=lambda storage, loc: storage
            )
            ckpt = ckpt['state_dict']
            modified_dict = {}
            for key, value in ckpt.items():
                if len(key) > 5:
                    new_key = key[6:]  
                    modified_dict[new_key] = value  
            result = self.model.load_state_dict(modified_dict)

            logging.info(f'load vsr pretrained model {self.cfg.pretrained_model}')
        else:
            logging.info('training vsr from scratch')



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "name": "model",
                    "params": self.model.parameters(),
                    "lr": self.cfg.optimizer.lr,
                }
            ],
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=(0.9, 0.98),
        )


        scheduler = WarmupCosineScheduler(
            optimizer,
            self.cfg.optimizer.warmup_epochs,
            self.cfg.trainer.max_epochs,
            len(self.trainer.datamodule.train_dataloader()),
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")


    
    def _step(self, batch, batch_idx, step_type):
        loss, loss_ctc, loss_att, acc = self.model(
            batch["inputs"], batch["input_lengths"], batch["targets"]
        )
        batch_size = len(batch["rel_path"])

        if step_type == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)
            self.log("train_loss_ctc", loss_ctc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)  
            self.log("train_loss_att", loss_att, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)                  
            self.log("train_decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)
        else:
            self.log("valid_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)
            self.log("valid_loss_ctc", loss_ctc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)  
            self.log("valid_loss_att", loss_att, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)      
            self.log("valid_decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)


        if step_type == "train":
            self.log(
                "monitoring_step", torch.tensor(self.global_step, dtype=torch.float32)
            )

        return loss

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.loaders.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

