import os
import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from sklearn.cluster import KMeans
from tqdm import tqdm
from joblib import dump
import pickle
import argparse

# Set CUDA device
device = 'cuda'

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Wav2Vec2 Feature Extraction and KMeans Clustering")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model directory')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing wav files')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the KMeans model')
    parser.add_argument('--cuda_visible_devices', type=str, default='0', help='CUDA device to use, default is 0')
    parser.add_argument('--num_files', type=int, default=10000, help='Number of audio files to process, default is 10000')
    parser.add_argument('--n_clusters', type=int, default=200, help='Number of clusters for KMeans, default is 200')
    return parser.parse_args()

# Main program
def main():
    args = parse_args()

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    # Load model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
    model = HubertModel.from_pretrained(args.model_path)
    model.to(device)
    model.half()  # Use half precision to speed up computation
    model.eval()

    # List to store all features
    all_features = []
    aaaa = []
    count = 0

    # Iterate through the folder and process each wav file
    for root, dirs, files in tqdm(os.walk(args.folder_path)):
        for file in tqdm(files, desc="Processing audio files", unit="file"):
            if file.endswith('.wav'):
                count += 1
                wav_path = os.path.join(root, file)
                wav, sr = sf.read(wav_path)
                aaaa.append(wav_path)
                input_values = feature_extractor(wav, return_tensors="pt", sampling_rate=sr).input_values
                input_values = input_values.half()
                input_values = input_values.to(device)

                with torch.no_grad():
                    outputs = model(input_values)
                    features = outputs.last_hidden_state.squeeze().cpu().numpy()  # Convert to numpy array
                    all_features.append(features)
            
            if count >= args.num_files:
                break
    
    print(f'Total number of files processed: {count}')

    # Stack all features into one large array
    all_features = np.vstack(all_features)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(all_features)

    # Save the model to file
    dump(kmeans, f'{args.save_path}/kmeans_model.joblib')

if __name__ == "__main__":
    main()