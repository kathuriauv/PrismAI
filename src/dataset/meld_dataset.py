import os
import csv
import torch
from torch.utils.data import Dataset
from .label_harmonizer import map_meld_label

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.video_dir = video_dir
        self.extractor = None
        self.data = []
        
        # Using pure Python CSV instead of Pandas to prevent Windows memory crashes
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                    'label': label_id,
                    'dataset_id': 1
                })
        print(f"Loaded {len(self.data)} valid samples from MELD.")

    def set_extractor(self, extractor):
        self.extractor = extractor

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
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'dataset_id': torch.tensor(sample['dataset_id'], dtype=torch.long)
        }