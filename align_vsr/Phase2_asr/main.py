import sys
sys.path.append('/work/liuzehua/task/VSR/cnvsrc')
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import hydra
import torch
from vsr2asr.model5.Phase2_asr.asr_dataset import DataModule
from vsr2asr.model5.Phase2_asr.lightning import ModelModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig


@hydra.main(config_path="/work/liuzehua/task/VSR/cnvsrc/conf/vsr2asr/model5/Phase2_asr", config_name= "train.yaml")
def main(cfg: DictConfig) -> None:
        seed_everything(42, workers=True)  #workers=True 指示 PyTorch Lightning 在初始化数据加载器的工作进程时也应该设置种子。
        cfg.gpus = torch.cuda.device_count()
    
        #make sure save path exists
        os.makedirs(cfg.save.save_path, exist_ok= True)
        os.makedirs(cfg.save.save_train_model, exist_ok = True)
        os.makedirs(cfg.save.save_valid_model, exist_ok = True)

        # monitor lr
        lr_monitor = LearningRateMonitor(logging_interval="step")

        #save checkpoint
        valid_checkpoint = ModelCheckpoint(
            monitor="train_loss",
            mode="min",
            dirpath=cfg.save.save_valid_model,
            save_last=True,
            filename="{epoch}-{train_loss:.2f}",
            save_top_k=cfg.save.save_valid_topk,
        )
        #save checkpoint
        train_checkpoint = ModelCheckpoint(
            monitor="valid_loss",
            mode="min",
            dirpath=cfg.save.save_train_model,
            filename="{epoch}-{valid_loss:.2f}",
            save_top_k=cfg.save.save_train_topk,
        )
        #save checkpoint
        decoder_acc_val = ModelCheckpoint(
            monitor="valid_decoder_acc",
            mode="max",
            dirpath=cfg.save.save_train_model,
            filename="{epoch}-{valid_decoder_acc:.4f}",
            save_top_k=cfg.save.save_train_topk,
        )
        callbacks = [valid_checkpoint, lr_monitor, train_checkpoint, decoder_acc_val]
        # Configure logger
        logger = TensorBoardLogger(save_dir=os.path.join(cfg.save.tblog_dir))
        dataset = DataModule(cfg)
        model = ModelModule(cfg)
        trainer = Trainer(
            **cfg.trainer,#相当于函数传参，只不过换了yaml的形式
            logger=logger,#这是记录模型的学习率等等
            strategy=DDPStrategy(find_unused_parameters=False) if cfg.gpus > 1 else None ,#分布式训练
            callbacks= callbacks,
        )

        # Training and testing

        trainer.fit(model=model, datamodule=dataset)

        # only 1 process should save the checkpoint and compute WER
        if cfg.gpus > 1:
            torch.distributed.destroy_process_group()

        if trainer.is_global_zero:#判断当前模型是不是主模型(训练结束后)
            #此处是训练结束后可选择保存模型的方法，以及一系列操作。
            pass


if __name__ == "__main__":
    main()

