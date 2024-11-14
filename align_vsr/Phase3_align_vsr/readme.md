# Training Configuration and Execution

This document provides an overview of the training configuration and execution setup for the VSR (Visual Speech Recognition) model in the `align_vsr/Phase3_align_vsr/conf/train.yaml` file and the associated execution script `bash run.sh`.

## Configuration File: `train.yaml`

### Parameters

- **csv_name**: `train`
  - The name of the CSV file used for training.
  
- **code_root_dir**: `/work/liuzehua/task/VSR/AlignVSR/align_vsr`
  - The root directory where the current code is located.

- **audio_data_root_dir**: `/work/liuzehua/task/VSR/data/LRS/LRS2-BBC`
  - The path where the audio data is stored.

- **video_data_root_dir**: `/work/liuzehua/task/VSR/data/LRS/LRS2-BBC`
  - The path where the video data is stored.

- **k_means_model**: `/work/liuzehua/task/VSR/cnvsrc/vsr2asr/model5/Phase1_k-means_cluster/kmeans_model.joblib`
  - The path to the pre-trained k-means model from Phase 1.

- **hubert_model**: `/work/liuzehua/task/VSR/cnvsrc/vsr2asr/model5/English-hubert-large`
  - The path to the pre-trained Hubert model (English version).

- **gpus**: `1`
  - Specifies the number of GPUs to be used for training.

- **pretrained_model**: `/work/liuzehua/task/VSR/cnvsrc/main_log/2024-09-10/21-09-54-model5-Phase3-vsr2asr-v2-lrs2-4s-model3/model/epoch=39-valid_asr2vsr_decoder_acc=0.4121.ckpt`
  - The path to the pre-trained model checkpoint for further fine-tuning.

- **CAM_path**: `/work/liuzehua/task/VSR/cnvsrc/main_log/2024-08-29/20-39-09-model5-Phase2-asr/model/epoch=79-train_loss=19.67.ckpt`
  - The path to the ASR (Automatic Speech Recognition) model trained in Phase 2.

### Loss Weights

- **loss.asr2vsr_att_w**: `0.9`
  - The weight for the attention loss in the ASR to VSR conversion.

- **loss.ctc_w**: `0.1`
  - The weight for the CTC (Connectionist Temporal Classification) loss.

- **loss.a2v_attscore.a2v_attscore_w**: `8`
  - The weight for the Align Loss (used for aligning audio and video features).

### Data Parameters

- **batch_max_frames**: `1000`
  - Maximum number of frames per batch.

- **max_frames**: `125`
  - The maximum number of frames for each video in the training set.

- **max_frames_val**: `1000`
  - The maximum number of frames for each video in the validation set.

- **dataset.root**: `${code_root_dir}`
  - Root directory for the dataset, set dynamically from the `code_root_dir`.

- **dataset.label_dir**: `/work/liuzehua/task/VSR/cnvsrc/data/vsr2asr/model5/Phase3/LRS2`
  - Directory where the CSV files containing the dataset labels are located.

## Execution Script: `run.sh`

### Overview

The `run.sh` script is used to run the training process for the model. It sets up the environment, defines paths for saving results, and triggers the training script.

### Script Details

```bash
#!/bin/bash

# Get the current date and time
date=$(date "+%Y-%m-%d")
time=$(date "+%H-%M-%S")

# Navigate to the training directory
cd /work/liuzehua/task/VSR/cnvsrc/vsr2asr/model5/Phase3_vsr2asr_v2

# Define the path to save results
save_path=/work/liuzehua/task/VSR/cnvsrc/main_log/$date/$time-model5-Phase3-vsr2asr-v2-lrs2-4s-model3

# Run the training script with specified GPU(s)
CUDA_VISIBLE_DEVICES=3,4,5 python main.py save.save_path=$save_path
