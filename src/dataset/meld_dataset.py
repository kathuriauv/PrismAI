import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .label_harmonizer import map_meld_label
from .feature_extractor import MultimodalFeatureExtractor

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir, max_text_len=128):
        self.video_dir = video_dir
        self.extractor = MultimodalFeatureExtractor(max_text_len=max_text_len)
        
        df = pd.read_csv(csv_path)
        
        self.data = []
        for _, row in df.iterrows():
            label_id = map_meld_label(row['Emotion'])
            
            if label_id is None:
                continue
            
            dia_id = row['Dialogue_ID']
            utt_id = row['Utterance_ID']
            
            wav_filename = f"dia{dia_id}_utt{utt_id}.wav"
            mp4_filename = f"dia{dia_id}_utt{utt_id}.mp4"
            
            wav_filepath = os.path.join(self.video_dir, wav_filename)
            mp4_filepath = os.path.join(self.video_dir, mp4_filename)
            
            self.data.append({
                'text': row['Utterance'],
                'wav_path': wav_filepath,
                'mp4_path': mp4_filepath if os.path.exists(mp4_filepath) else None,
                'label': label_id
            })
            
        print(f"Loaded {len(self.data)} valid samples from MELD.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        text_features = self.extractor.process_text(sample['text'])
        audio_features = self.extractor.process_audio(sample['wav_path'])
        
        if sample['mp4_path'] is not None:
            video_frame = self.extractor.process_video(sample['mp4_path'])
        else:
            video_frame = torch.zeros(3, 224, 224)
        
        return {
            'input_ids': text_features['input_ids'],
            'attention_mask': text_features['attention_mask'],
            'audio_features': audio_features,
            'video_frame': video_frame,
            'label': sample['label']
        }

if __name__ == "__main__":
    test_csv = r"C:\Users\kathu\OneDrive\Desktop\Projects\PrismAI_v1\data\raw\MELD-RAW\MELD.Raw\train\train_sent_emo.csv"
    test_vid = r"C:\Users\kathu\OneDrive\Desktop\Projects\PrismAI_v1\data\raw\MELD-RAW\MELD.Raw\train\train_splits"
    
    dataset = MELDDataset(csv_path=test_csv, video_dir=test_vid)
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nFINAL TENSOR SHAPES")
        print(f"Text Input IDs Shape: {sample['input_ids'].shape}")
        print(f"Audio Spectrogram Shape: {sample['audio_features'].shape}")
        print(f"Video Frame Shape: {sample['video_frame'].shape}")
        print(f"Label: {sample['label']}")