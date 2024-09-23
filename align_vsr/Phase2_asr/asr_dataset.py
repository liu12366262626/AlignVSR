import sys
sys.path.append('/work/liuzehua/task/VSR/cnvsrc')
import os
import logging
import torch
from pytorch_lightning import LightningDataModule
from vsr2asr.model5.Phase2_asr.samplers import (
    ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)
from vsr2asr.model5.Phase2_asr.transforms import AudioTransform, VideoTransform
import torchaudio
import torchvision
from espnet.nets.pytorch_backend.backbones.conv1d_extractor import Conv1dResNet


from espnet.nets.pytorch_backend.transformer.embedding import (
    RelPositionalEncoding,  # noqa: H301
)

import soundfile as sf
from transformers import (
    Wav2Vec2FeatureExtractor,
)

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
        if data_type == 'audio_rel_path' or data_type == 'video_rel_path' or data_type == 'rel_path':
            batch_out[data_type] = [d[data_type] for d in batch if data_type in d]
            continue
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


def cut_or_pad(data, size, audio_path, dim=0):
    #42864 + 16 = 42880
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    
    return data


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)



class ASR_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_data_root,
        label_path,
        subset,
        audio_transform,
        cfg,
        rate_ratio=640,
        max_frame=1800,
    ):

        self.audio_data_root = audio_data_root
        self.rate_ratio = rate_ratio
        self.max_frame = int(max_frame)

        self.list = self.load_list(label_path)
        logging.info(f'{subset} dataset load data {len(self.list)}')
        self.audio_transform = audio_transform
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.hubert_model)

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            if int(input_length) < self.max_frame :
                path = os.path.join(self.audio_data_root, dataset_name, rel_path)
                if os.path.exists(path):
                    token = [int(_) for _ in token_id.split()]
                    if len(token) < 1:
                        continue
                    paths_counts_labels.append(
                        (
                            dataset_name,
                            rel_path,
                            int(input_length),
                            torch.tensor([int(_) for _ in token_id.split()]),
                        )
                    )
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.audio_data_root, dataset_name, rel_path)


        wav, sr = sf.read(path)
        audio =self.feature_extractor(wav, return_tensors="pt", sampling_rate = sr).input_values
        audio = audio.transpose(1, 0)

        return {"input": audio, "target": token_id, 'rel_path': rel_path}

    def __len__(self):
        return len(self.list)




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
        train_ds = ASR_Dataset(
            audio_data_root = self.cfg.audio_data_root_dir,
            label_path=os.path.join(
                ds_args.root, ds_args.label_dir, ds_args.train_file
            ),
            subset="train",
            audio_transform=AudioTransform("train"),
            cfg = self.cfg,
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
        val_ds = ASR_Dataset(
            audio_data_root = self.cfg.audio_data_root_dir,
            label_path=os.path.join(ds_args.root, ds_args.label_dir, ds_args.val_file),
            subset="val",
            audio_transform=AudioTransform("val"),
            cfg = self.cfg,
            max_frame = self.cfg.data.max_frames_val,
        )
        sampler = ByFrameCountSampler(
            val_ds, self.cfg.data.batch_max_frames, shuffle=False
        )
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        return self._dataloader(val_ds, sampler, collate_pad)

