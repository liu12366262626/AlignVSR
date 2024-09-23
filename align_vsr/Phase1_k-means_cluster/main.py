import os
import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from sklearn.cluster import KMeans
from tqdm import tqdm
from joblib import dump
# from cuml.cluster import KMeans
import pickle
# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = 'cuda'

# 加载模型
model_path = "/work/liuzehua/task/VSR/cnvsrc/vsr2asr/model5/English-hubert-large"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = HubertModel.from_pretrained(model_path)
model.to(device)
model.half()  # 使用半精度加快运算速度
model.eval()

# 指定文件夹路径
folder_path = '/work/liuzehua/task/VSR/data/LRS/LRS2-BBC/lrs2/lrs2_video_seg24s/pretrain'

# 存储所有特征的列表
all_features = []
aaaa = []
count = 0
# 遍历文件夹处理每个wav文件
for root, dirs, files in tqdm(os.walk(folder_path)):
    for file in tqdm(files, desc="Processing audio files", unit="file"):
        if file.endswith('.wav'):
            count = count + 1
            wav_path = os.path.join(root, file)
            wav, sr = sf.read(wav_path)
            aaaa.append(wav_path)
            input_values = feature_extractor(wav, return_tensors="pt", sampling_rate=sr).input_values
            input_values = input_values.half()
            input_values = input_values.to(device)

            with torch.no_grad():
                outputs = model(input_values)
                features = outputs.last_hidden_state.squeeze().cpu().numpy()  # 转为numpy数组
                all_features.append(features)
    print(f'count_num: {count}')
    if count >= 20000:
        break

print(len(all_features))

# 将所有特征堆叠成一个大数组
all_features = np.vstack(all_features)

# 执行k-means聚类
kmeans = KMeans(n_clusters=500, random_state=0).fit(all_features)


save_path = '/work/liuzehua/task/VSR/cnvsrc/vsr2asr/model5/Phase1_k-means_cluster'

# 保存模型到文件
dump(kmeans, f'{save_path}/kmeans_model_lrs2_500class_20k.joblib')