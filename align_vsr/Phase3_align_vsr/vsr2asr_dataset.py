import sys
sys.path.append('./align_vsr')
import os
import logging
import torch
from pytorch_lightning import LightningDataModule
from align_vsr.Phase3_align_vsr.samplers import (
    ByFrameCountSampler,
    DistributedSamplerWrapper,
    RandomSamplerWrapper,
)
from align_vsr.Phase3_align_vsr.transforms import AudioTransform, VideoTransform
import torchaudio
import torchvision
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel
)

import soundfile as sf
from joblib import load
import random
import csv
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
        if data_type == 'audio_rel_path' or data_type == 'video_rel_path'or data_type == "audio_label" :
            batch_out[data_type] = [d[data_type] for d in batch if data_type in d]
            continue
        pad_val = -1 if data_type == "target" else 0.0
        c_batch, sample_lengths = pad(
            [s[data_type] for s in batch if s[data_type] is not None], pad_val
        )
        batch_out[data_type + "s"] = c_batch
        batch_out[data_type + "_lengths"] = torch.tensor(sample_lengths)
    return batch_out


def cut_or_pad(data, size, dim=0):
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



class V2A_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_data_root,
        video_data_root,
        label_path,
        subset,
        audio_transform,
        video_transform,
        cfg,
        rate_ratio=640,
        max_frame=1800,
    ):

        self.audio_data_root = audio_data_root
        self.video_data_root = video_data_root
        self.rate_ratio = rate_ratio
        self.max_frame = int(max_frame)

        self.list = self.load_list(label_path)
        logging.info(f'{subset} dataset load data {len(self.list)}')
        self.audio_transform = audio_transform
        self.video_transform = video_transform


        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.hubert_model)
        self.hubert_model = HubertModel.from_pretrained(cfg.hubert_model)
        self.kmeans = load(cfg.k_means_model)

    def load_list(self, label_path):
        paths_counts_labels = []
        with open(label_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                dataset_name, video_rel_path, audio_rel_path, input_length, token_id, audio_label = row
                audio_label = eval(audio_label)
                if int(input_length) < self.max_frame :
                    video_path = os.path.join(self.video_data_root, dataset_name, video_rel_path)
                    audio_path = os.path.join(self.audio_data_root, dataset_name, audio_rel_path)
                    if os.path.exists(video_path) and os.path.exists(audio_path):
                        token = [int(_) for _ in token_id.split()]
                        if len(token) < 1:
                            continue
                        paths_counts_labels.append(
                            (
                                dataset_name,
                                video_rel_path,
                                audio_rel_path,
                                int(input_length),
                                torch.tensor([int(_) for _ in token_id.split()]),
                                audio_label,
                            )
                        )
            return paths_counts_labels
    
    def __getitem__(self, idx):
        dataset_name, video_rel_path, audio_rel_path, input_length, token_id, audio_label = self.list[idx]
        video_path = os.path.join(self.video_data_root, dataset_name, video_rel_path)
        audio_path = os.path.join(self.audio_data_root, dataset_name, audio_rel_path)
        video = load_video(video_path)
        video = self.video_transform(video)
        labels_tensor = torch.tensor(audio_label)

        # wav, sr = sf.read(audio_path)
        # input_values = self.feature_extractor(wav, return_tensors="pt", sampling_rate = sr).input_values
        # cut_or_pad(input_values.transpose(1,0), len(video) * self.rate_ratio, )
        # input_values = input_values.to(video.device)
        # outputs = self.hubert_model(input_values).last_hidden_state

        # # Reshape x to (batch_size * time_steps, embedding_size)
        # batch_size, time_steps, embedding_size = outputs.size()
        # x_reshaped = outputs.view(batch_size * time_steps, embedding_size).detach().cpu()

        # # Predict cluster center indices for each sample using the k-means model
        # labels = self.kmeans.predict(x_reshaped)

        # # Convert labels to a PyTorch tensor and ensure it is on the same device as the original tensor
        # labels_tensor = torch.tensor(labels, dtype=torch.long, device=outputs.device).view(batch_size, time_steps)
        # labels_tensor = labels_tensor.squeeze(0)
        # # labels_tensor = torch.randint(low=0, high=200, size=(random.randint(0, 199),))



        return {"video": video, "audio_label": labels_tensor,"target": token_id, 'video_rel_path': video_rel_path}

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
        train_ds = V2A_Dataset(
            audio_data_root = self.cfg.audio_data_root_dir,
            video_data_root = self.cfg.video_data_root_dir,
            label_path=os.path.join(
                ds_args.root, ds_args.label_dir, ds_args.train_file
            ),
            subset="train",
            audio_transform=AudioTransform("train"),
            video_transform=VideoTransform("train"),
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
        val_ds = V2A_Dataset(
            audio_data_root = self.cfg.audio_data_root_dir,
            video_data_root = self.cfg.video_data_root_dir,
            label_path=os.path.join(ds_args.root, ds_args.label_dir, ds_args.val_file),
            subset="val",
            audio_transform=AudioTransform("val"),
            video_transform=VideoTransform("val"),
            cfg = self.cfg,
            max_frame = self.cfg.data.max_frames_val,
        )
        sampler = ByFrameCountSampler(
            val_ds, self.cfg.data.batch_max_frames, shuffle=False
        )
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        return self._dataloader(val_ds, sampler, collate_pad)

