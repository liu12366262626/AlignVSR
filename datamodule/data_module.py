import os
import logging
import torch
from pytorch_lightning import LightningDataModule

from .av_dataset import AVDataset
from .samplers import (
    ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)
from .transforms import AudioTransform, VideoTransform


# https://github.com/facebookresearch/av_hubert/blob/593d0ae8462be128faab6d866a3a926e2955bde1/avhubert/hubert_dataset.py#L517
def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) == 1:
        collated_batch = collated_batch.unsqueeze(1)  # targets
    elif len(samples[0].shape) == 2:
        pass  # collated_batch: [B, T, 1]
    elif len(samples[0].shape) == 4:
        pass  # collated_batch: [B, T, C, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in batch[0].keys():
        if data_type == 'rel_path':
            batch_out['rel_path'] = [d['rel_path'] for d in batch if 'rel_path' in d]
            continue
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


class DataModule(LightningDataModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.cfg.gpus = torch.cuda.device_count()
        self.total_gpus = self.cfg.gpus 

    def _dataloader(self, ds, sampler, collate_fn):


        return torch.utils.data.DataLoader(
            ds,
            num_workers=self.cfg.data.dataset.num_workers,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        ds_args = self.cfg.data.dataset
        train_ds = AVDataset(
            root=self.cfg.data_root_dir,
            label_path=os.path.join(
                ds_args.root, ds_args.label_dir, ds_args.train_file
            ),
            subset="train",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("train"),
            video_transform=VideoTransform("train", self.cfg),
            max_frame = self.cfg.data.max_frames,
        )
        sampler = ByFrameCountSampler(train_ds, self.cfg.data.batch_max_frames)#这个sampler的batch是以frame去计算的
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = RandomSamplerWrapper(sampler)
        return self._dataloader(train_ds, sampler, collate_pad)

    def val_dataloader(self):
        ds_args = self.cfg.data.dataset
        val_ds = AVDataset(
            root=self.cfg.data_root_dir,
            label_path=os.path.join(ds_args.root, ds_args.label_dir, ds_args.val_file),
            subset="val",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform("val"),
            video_transform=VideoTransform("val", self.cfg),
            max_frame = self.cfg.data.max_frames_val,
        )
        sampler = ByFrameCountSampler(
            val_ds, self.cfg.data.batch_max_frames, shuffle=False
        )
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        return self._dataloader(val_ds, sampler, collate_pad)

    def test_dataloader(self):
        ds_args = self.cfg.data.dataset
        dataset = AVDataset(
            root=self.cfg.data_root_dir,
            label_path=os.path.join(ds_args.root, ds_args.label_dir, ds_args.test_file),
            subset="test",
            modality=self.cfg.data.modality,
            audio_transform=AudioTransform(
                "test", snr_target=self.cfg.decode.snr_target
            ),
            video_transform=VideoTransform("test", self.cfg),
            max_frame = self.cfg.data.max_frames,
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
        return dataloader
