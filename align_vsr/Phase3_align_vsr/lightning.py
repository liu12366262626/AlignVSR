import sys
sys.path.append('/work/liuzehua/task/VSR/cnvsrc')
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from cosine import WarmupCosineScheduler
from vsr2asr.model5.Phase3_vsr2asr_v2.transforms import TextTransform
import logging

# for testing
from espnet.asr.asr_utils import add_results_to_json
from pytorch_lightning import LightningModule
from vsr2asr.model5.Phase3_vsr2asr_v2.vsr2asr_model import V2A
import os



class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.hp_metric = {
        'valid_min_ctc': 100000000000.0,
        'valid_min_att': 100000000000.0,
        'valid_min_attscore': 100000000000.0,
        'valid_max_decoder_acc': 0.0,
        }
        self.cfg = cfg

        self.backbone_args = self.cfg.model.visual_backbone

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = V2A(len(self.token_list), cfg)

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

    def forward(self, sample):
        # self.beam_search = get_beam_search_decoder(self.model, self.token_list, ctc_weight=self.backbone_args.mtlalpha, lm_weight=self.cfg.lm_weight)
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
        # enc_feat = enc_feat.squeeze(0)
        # nbest_hyps = self.beam_search(enc_feat)
        # nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        # predicted = add_results_to_json(nbest_hyps, self.token_list)
        # predicted = predicted.replace("▁", " ").strip().replace("<eos>", "")
        return None

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")


    
    def _step(self, batch, batch_idx, step_type):
        loss, asr2vsr_loss_att, asr2vsr_acc, loss_ctc, a2v_attscore_loss = self.model(
            batch["videos"], batch["video_lengths"], batch["targets"], batch['audio_label']
        )
        batch_size = batch["videos"].size(0)

        if step_type == "train":
            self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)
            self.log("train_loss_ctc", loss_ctc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)  
            self.log("train_loss_asr2vsr_att", asr2vsr_loss_att, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)                  
            self.log("train_asr2vsr_decoder_acc", asr2vsr_acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)  
            self.log("train_a2v_attscore_loss", a2v_attscore_loss, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)         
            # self.log("train_loss_att_reversed", loss_att_reversed, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)  
            # self.log("train_acc_reversed", acc_reversed, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)  
        else:
            self.log("valid_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)
            self.log("valid_loss_ctc", loss_ctc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)  
            self.log("valid_loss_asr2vsr_att", asr2vsr_loss_att, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)                  
            self.log("valid_asr2vsr_decoder_acc", asr2vsr_acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)    
            self.log("valid_a2v_attscore_loss", a2v_attscore_loss, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)   
            # self.log("valid_loss_att_reversed", loss_att_reversed, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)  
            # self.log("valid_acc_reversed", acc_reversed, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist= True)  
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

    def on_train_epoch_end(self):
        
        self.hp_metric['valid_min_ctc'] = min(self.hp_metric['valid_min_ctc'], float(self.trainer.callback_metrics['valid_loss_ctc_epoch']))
        self.hp_metric['valid_min_att'] = min(self.hp_metric['valid_min_att'], float(self.trainer.callback_metrics['valid_loss_asr2vsr_att_epoch'])) 
        self.hp_metric['valid_min_attscore'] = min(self.hp_metric['valid_min_attscore'], float(self.trainer.callback_metrics['valid_a2v_attscore_loss_epoch']))       
        self.hp_metric['valid_max_decoder_acc'] = max(self.hp_metric['valid_max_decoder_acc'], float(self.trainer.callback_metrics['valid_asr2vsr_decoder_acc_epoch']))

    
    def on_train_end(self):
        cfg = self.cfg
        hparams = {
            "model_name": os.path.basename(cfg.save.save_path),
            "train_data": cfg.csv_name,
            "loss": cfg.loss,
            "oprimizer": cfg.optimizer,
            "epoch": cfg.trainer.max_epochs
            # 可以添加更多的超参数
        }
        self.logger.log_hyperparams(hparams, self.hp_metric)


