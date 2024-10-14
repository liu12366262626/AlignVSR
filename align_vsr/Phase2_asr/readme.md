# ASR Model Training Script

This repository contains a script for training an Automatic Speech Recognition (ASR) model. The script includes customizable options for various paths and models, allowing flexible training configuration.

## Script Overview

The provided `run.sh` script is used to initiate the training process of an ASR model. The script sets up necessary environment variables and directories, and runs the training using specific audio data, models, and configurations.

### Key Parameters

- **`save.save_path`**: Specifies the path where the final training results will be saved. The path includes both the date and time of the training for easier result tracking.

- **`audio_data_root_dir`**: The directory where the raw audio data is stored. This data will be used during training.

- **`csv_name`**: The name of the CSV file containing training data, specifying how the audio data is organized.

- **`hubert_model`**: The pre-trained HuBERT model path used in the ASR task for feature extraction.

- **`k_means_model`**: The K-means model obtained from the first phase of training, which is used as part of the feature extraction pipeline in the second phase of ASR training.

- **`data.dataset.label_dir`**: The directory where the labels (e.g., transcriptions) for the dataset are stored. This folder includes files like `train.csv` and `test.csv`, which define the labels for training and evaluation.

- **`data.dataset.val_file`**: Specifies the CSV file used for the validation set, allowing the model to be evaluated on a separate dataset during training.

### Example Usage

Here’s an example command to run the ASR training:

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py save.save_path=$save_path \
                        code_root_dir=$code_root_dir\
                        audio_data_root_dir=/path/to/audio/data \
                        csv_name=train \
                        hubert_model=/path/to/hubert_model \
                        k_means_model=/path/to/k_means_model.joblib \
                        data.dataset.label_dir=/path/to/label/directory \
                        data.dataset.val_file=test.csv
