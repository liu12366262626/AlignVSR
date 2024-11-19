"This repository is directly related to the AlignVSR paper. We will continue to maintain and improve the code in the future."

<div align="center">
    <img src="example.gif" width="256" />
</div>

# Data Strcuture

LRS2-BBC
├── labels
├── lrs2
│   ├── lrs2_text_seg24s
│   │   ├── main
│   │   ├── pretrain
│   ├── lrs2_video_seg24s
├── LRS2-TED


# Preprocess
We have adopted a consistent approach with the [AUTO-AVSR repository](https://github.com/mpc001/auto_avsr) for preprocessing the LRS2 and Single datasets.
Then, following the steps from [AUTO-AVSR (preparation)](https://github.com/mpc001/auto_avsr/tree/main/preparation), we process the LRS2 and CNVSRC.Single datasets to generate the corresponding `train.csv` and `test.csv`.

# Phase1-K-means
For the LRS2 and CNVSRC.Single datasets, we randomly sample a portion of the audio data from the training set to train a k-means model with a total of 200 clusters. For specific steps, please refer to [this link](https://github.com/liu12366262626/AlignVSR/tree/master/align_vsr/Phase1_k-means_cluster). After completing this step, we will obtain the k-means model for the next phase of training.

# Phase2-ASR-Training
We use the pre-trained [Hubert model](https://huggingface.co/facebook/hubert-large-ll60k) and the trained k-means model to quantize the audio data. For the quantized audio, we use Conformer as the Encoder and train it in an ASR paradigm with the hybrid CTC/Attention Loss. For detailed steps, please refer to [this link](https://github.com/liu12366262626/AlignVSR/tree/master/align_vsr/Phase2_asr).

# Phase3-AlignVSR
After completing Phase2, we use the obtained quantized audio as the K (Key) and V (Value) in the Cross-Attention mechanism. The video features are used as Q (Query) and are input into the Cross-Attention. Additionally, we introduce the Local Align Loss to align the audio and video features at the frame level. For detailed steps, please refer to [this link](https://github.com/liu12366262626/AlignVSR/tree/master/align_vsr/Phase3_align_vsr).