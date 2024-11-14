## Dataset Directory Structure
dataset/ ├── Phase2/ │ ├── test.csv │ └── train.csv ├── Phase3/ │ ├── test.csv │ └── train.csv

# Dataset Preparation for Training

This repository provides example training (`train.csv`) and testing (`test.csv`) files for both Phase 2 and Phase 3 of the Visual Speech Recognition (VSR) task. By organizing all data from the LRS2 dataset into the specified format, you will be able to proceed with model training.

## CSV File Format: `train.csv`

The `train.csv` file is structured as follows:


Column Descriptions
Column 1: Dataset identifier (lrs2).
Column 2: Relative path to the video file, e.g., lrs2_video_seg24s/main/5535415699068794046/00007.mp4.
Column 3: Relative path to the corresponding audio file, e.g., lrs2_video_seg24s/main/5535415699068794046/00007.wav.
Column 4: Number of video frames in the sample, e.g., 71.
Column 5: Label sequence for the video, represented as a sequence of numbers. This sequence is generated using the unigram5000_units.txt file located in AlignVSR/align_vsr/English_unigram/, which maps words to numeric labels.
Column 6: Frame-level audio alignment indices represented as an array. This sequence is generated using preprocess_asr_label.py from the AlignVSR/align_vsr/Phase3_align_vsr/ directory, and it provides frame-level correspondence between audio and video.