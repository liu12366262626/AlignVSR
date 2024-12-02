"This repository is directly related to the AlignVSR paper. We will continue to maintain and improve the code in the future."

<div align="center">
    <img src="example.gif" width="256" />
</div>

# Data Strcuture

```
LRS2-BBC
├── lrs2
│   ├── lrs2_video_seg24s
│   │   ├── main
|   │   │   ├── 5535415699068794046
|   |   │   │   ├── 0001.mp4
|   |   │   │   ├── 0001.wav
|   |   │   │   ├── ...
|   │   │   ├── ...
│   │   ├── pretrain
|   │   │   ├── 5535415699068794046
|   |   │   │   ├── 00001.mp4
|   |   │   │   ├── 00001.wav
|   |   │   │   ├── ...
|   │   │   ├── ...
```


# 1. Environment Setup and Preprocess
We have adopted a consistent approach with the [AUTO-AVSR repository](https://github.com/mpc001/auto_avsr) for preprocessing the LRS2 and Single datasets.
Then, following the steps from [AUTO-AVSR (preparation)](https://github.com/mpc001/auto_avsr/tree/main/preparation), we process the LRS2 and CNVSRC.Single datasets to generate the corresponding `train.csv` and `test.csv`.

## 1.1 AlignVSR Environment Setup

This guide will walk you through the process of setting up the `AlignVSR` environment, installing necessary dependencies, and preparing for preprocessing.

```
git clone git@github.com:liu12366262626/AlignVSR.git
conda env create -f alignvsr_env.yaml 
conda activate alignvsr
cd tools/face_alignment 
pip install --editable .
cd tools/face_detection
pip install --editable .
```

## 1.2 Preprocess

Preprocess Dataset
```
cd preprocess_data
python preprocess.py --root_dir /[path-to-origin_LRS2_data] --dst_path /[path-to-save-preprocess_data]
```
- `--root_dir` /path/to/LRS2: Specifies the path to the input dataset (LRS2).
- `--dst_path` /path/to/preprocess2: Specifies the path where the preprocessed data should be stored.

After preprocessing all the video files, you need to generate corresponding audio files,the code is:

```
cd preprocess_data
python generate_audio.py --root_dir /[path-to-origin_LRS2_data]  --dst_path /[path-to-save-preprocess_data]/data
```
- `--root_dir` /path/to/LRS2: Specifies the path to the input dataset (LRS2).
- `--dst_path` /path/to/preprocess2: Specifies the path where the audio data should be stored.


Finally , the videos will be processed into a size of 96x96, at 25fps, and the audio will be processed into mono with a 16k sample rate.
# 2. Phase1-K-means
For the LRS2 and CNVSRC.Single datasets, we randomly sample a portion of the audio data from the training set to train a k-means model with a total of 200 clusters. For specific steps, please refer to [this link](https://github.com/liu12366262626/AlignVSR/tree/master/align_vsr/Phase1_k-means_cluster). After completing this step, we will obtain the k-means model for the next phase of training.

# 3. Phase2-ASR-Training
We use the pre-trained [Hubert model](https://huggingface.co/facebook/hubert-large-ll60k) and the trained k-means model to quantize the audio data. For the quantized audio, we use Conformer as the Encoder and train it in an ASR paradigm with the hybrid CTC/Attention Loss. For detailed steps, please refer to [this link](https://github.com/liu12366262626/AlignVSR/tree/master/align_vsr/Phase2_asr).

# 4. Phase3-AlignVSR
After completing Phase2, we use the obtained quantized audio as the K (Key) and V (Value) in the Cross-Attention mechanism. The video features are used as Q (Query) and are input into the Cross-Attention. Additionally, we introduce the Local Align Loss to align the audio and video features at the frame level. For detailed steps, please refer to [this link](https://github.com/liu12366262626/AlignVSR/tree/master/align_vsr/Phase3_align_vsr).
