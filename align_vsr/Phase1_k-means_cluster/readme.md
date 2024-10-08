# Phase1 K-Means

## Overview

This script is used to extract features from `.wav` audio files and perform KMeans clustering on these features. The main steps include:

1. **Load Pre-trained Model**: Use the Hugging Face `HubertModel` (https://huggingface.co/facebook/hubert-large-ll60k) as the pre-trained model.
2. **Iterate Through Audio Files**: Iterate through all `.wav` files in the specified folder and use the pre-trained model to extract features from these audio files.
3. **Feature Extraction**: Extract features from the audio files and stack them into a large array for further clustering analysis.
4. **KMeans Clustering**: Perform KMeans clustering on the extracted features to help identify similarities between the features.
5. **Save the Cluster Model**: Save the trained KMeans model to the specified path for future use.

## Parameters

The script uses command-line arguments to specify input/output paths and other parameters. Here is a detailed description of each parameter:

- `--model_path`: The directory path of the pre-trained model (required). Specifies the location of the HubertModel.
- `--folder_path`: The folder containing `.wav` files to be processed (required). Specifies the folder where audio files are located for feature extraction.
- `--save_path`: The path to save the KMeans model (required). Specifies where to save the cluster model.
- `--cuda_visible_devices`: The CUDA device number (optional). The default value is `0`, which specifies which GPU to use for computation.
- `--num_files`: The number of audio files to process (optional). The default value is `10000`, which limits the number of audio files to be processed.
- `--n_clusters`: The number of clusters for KMeans (optional). The default value is `200`, which specifies the number of clusters for the clustering analysis.

## Example Usage

Assuming the pre-trained model is located at `AlignVSR/checkpoints/English-hubert-large`, audio files are located at `[your lrs2 path]/lrs2/lrs2_video_seg24s/pretrain`, and we want to save the KMeans model to `AlignVSR/checkpoints/k-means`, you can run the following command:

```bash
python wav2vec2_clustering.py \
    --model_path AlignVSR/checkpoints/English-hubert-large \
    --folder_path [your lrs2 path]/lrs2/lrs2_video_seg24s/pretrain \
    --save_path AlignVSR/checkpoints/k-means \
    --cuda_visible_devices 0 \
    --num_files 10000 \
    --n_clusters 200
```

## Notes

- Make sure the `model_path` and `folder_path` provided are valid, otherwise, the script will not run properly.
- If computational resources are limited, you can reduce `num_files` to decrease the workload.
- When using a GPU for computation, ensure that the GPU device number is set correctly to fully utilize GPU acceleration.

