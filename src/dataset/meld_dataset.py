import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .label_harmonizer import map_meld_label
from .feature_extractor import MultimodalFeatureExtractor

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir, max_text_len=128):
        """
        The Librarian for the MELD Dataset.
        Now equipped with the MultimodalFeatureExtractor!
        """
        self.video_dir = video_dir
        
        # Initialize translator
        self.extractor = MultimodalFeatureExtractor(max_text_len=max_text_len)
        
        # 1. Read the CSV file
        df = pd.read_csv(csv_path)
        
        # 2. Clean the data
        self.data = []
        for index, row in df.iterrows():
            raw_emotion = row['Emotion']
            label_id = map_meld_label(raw_emotion)
            
            if label_id is None:
                continue
                
            dia_id = row['Dialogue_ID']
            utt_id = row['Utterance_ID']
            filename = f"dia{dia_id}_utt{utt_id}.wav"
            filepath = os.path.join(self.video_dir, filename)
            
            self.data.append({
                'text': row['Utterance'],
                'filepath': filepath,
                'label': label_id
            })
            
        print(f"Loaded {len(self.data)} valid samples from MELD.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Get the raw text and file path
        sample = self.data[idx]
        
        # 2. Translate Text to RoBERTa IDs
        text_features = self.extractor.process_text(sample['text'])
        
        # 3. Translate Audio to Log-Mel Spectrograms
        
        audio_features = self.extractor.process_audio(sample['filepath'])
        
        # 4. Return the fully processed dictionary
        return {
            'input_ids': text_features['input_ids'],
            'attention_mask': text_features['attention_mask'],
            'audio_features': audio_features,
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }

if __name__ == "__main__":
    # TEST BLOCK 
    test_csv = r"C:\Users\kathu\OneDrive\Desktop\Projects\PrismAI_v1\data\raw\MELD-RAW\MELD.Raw\train\train_sent_emo.csv"
    test_vid = r"C:\Users\kathu\OneDrive\Desktop\Projects\PrismAI_v1\data\raw\MELD-RAW\MELD.Raw\train\train_splits"
    
    print("Initializing MELD Dataset with Feature Extractor (This will take a few seconds)...")
    dataset = MELDDataset(csv_path=test_csv, video_dir=test_vid)
    
    print("\nExtracting Sample 0...")
    sample_data = dataset[0]
    
    print("\nFINAL TENSOR SHAPES ")
    print(f"Text Input IDs Shape: {sample_data['input_ids'].shape} (Should be 128)")
    print(f"Audio Spectrogram Shape: {sample_data['audio_features'].shape} (Should be [1, 64, X])")
    print(f"Label: {sample_data['label']}")