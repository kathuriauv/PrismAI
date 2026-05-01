import os
import re
import torch
from torch.utils.data import Dataset
from .label_harmonizer import map_iemocap_label
from .feature_extractor import MultimodalFeatureExtractor

class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        self.extractor = MultimodalFeatureExtractor()
        
        sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        
        for session in sessions:
            session_path = os.path.join(self.data_dir, session)
            if not os.path.exists(session_path):
                continue
                
            eval_dir = os.path.join(session_path, 'dialog', 'EmoEvaluation')
            trans_dir = os.path.join(session_path, 'dialog', 'transcriptions')
            wav_base_dir = os.path.join(session_path, 'sentences', 'wav')
            
            for eval_file in os.listdir(eval_dir):
                if not eval_file.endswith('.txt'):
                    continue
                    
                eval_path = os.path.join(eval_dir, eval_file)
                base_name = eval_file.replace('.txt', '')
                
                trans_path = os.path.join(trans_dir, eval_file)
                if not os.path.exists(trans_path):
                    continue
                
                transcriptions = {}
                with open(trans_path, 'r') as tf:
                    for line in tf:
                        parts = line.strip().split(']: ')
                        if len(parts) == 2:
                            utt_id = parts[0].split(' ')[0]
                            text = parts[1]
                            transcriptions[utt_id] = text

                with open(eval_path, 'r') as ef:
                    for line in ef:
                        if not line.startswith('['):
                            continue
                            
                        parts = re.split(r'\t+', line.strip())
                        if len(parts) >= 3:
                            utt_id = parts[1]
                            raw_emotion = parts[2]
                            
                            label_id = map_iemocap_label(raw_emotion)
                            
                            if label_id is not None and utt_id in transcriptions:
                                wav_path = os.path.join(wav_base_dir, base_name, f"{utt_id}.wav")
                                
                                if os.path.exists(wav_path):
                                    self.data.append({
                                        'text': transcriptions[utt_id],
                                        'filepath': wav_path,
                                        'label': label_id
                                    })
                                    
        print(f"Loaded {len(self.data)} valid samples from IEMOCAP (Discarded unmapped emotions).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text_features = self.extractor.process_text(sample['text'])
        audio_features = self.extractor.process_audio(sample['filepath'])
        return {
            'input_ids': text_features['input_ids'],
            'attention_mask': text_features['attention_mask'],
            'audio_features': audio_features,
            'label': sample['label']
        }

if __name__ == "__main__":
    test_iemocap = r"C:\Users\kathu\OneDrive\Desktop\Projects\PrismAI_v1\data\raw\IEMOCAP"
    
    dataset = IEMOCAPDataset(data_dir=test_iemocap)
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nFINAL TENSOR SHAPES")
        print(f"Text Input IDs Shape: {sample['input_ids'].shape}")
        print(f"Audio Spectrogram Shape: {sample['audio_features'].shape}")
        print(f"Label: {sample['label']}")