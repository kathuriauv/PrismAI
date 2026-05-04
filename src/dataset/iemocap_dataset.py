import os
import re
import torch
from torch.utils.data import Dataset
from .label_harmonizer import map_iemocap_label

class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = []
        self.extractor = None

        sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

        for session in sessions:
            session_path = os.path.join(self.data_dir, session)
            if not os.path.exists(session_path):
                continue

            eval_dir = os.path.join(session_path, 'dialog', 'EmoEvaluation')
            trans_dir = os.path.join(session_path, 'dialog', 'transcriptions')
            wav_base_dir = os.path.join(session_path, 'sentences', 'wav')
            avi_dir = os.path.join(session_path, 'dialog', 'avi', 'DivX')

            for eval_file in os.listdir(eval_dir):
                if not eval_file.endswith('.txt'):
                    continue

                eval_path = os.path.join(eval_dir, eval_file)
                base_name = eval_file.replace('.txt', '')

                trans_path = os.path.join(trans_dir, eval_file)
                if not os.path.exists(trans_path):
                    continue

                avi_path = os.path.join(avi_dir, f"{base_name}.avi")

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

                        time_match = re.match(r'\[(\d+\.\d+) - (\d+\.\d+)\]', line)
                        if not time_match:
                            continue

                        start_time = float(time_match.group(1))
                        end_time = float(time_match.group(2))

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
                                        'wav_path': wav_path,
                                        'avi_path': avi_path if os.path.exists(avi_path) else None,
                                        'start_time': start_time,
                                        'end_time': end_time,
                                        'label': label_id,
                                        'dataset_id': 0
                                    })

        print(f"Loaded {len(self.data)} valid samples from IEMOCAP.")

    def set_extractor(self, extractor):
        self.extractor = extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        text_features = self.extractor.process_text(sample['text'])
        audio_features = self.extractor.process_audio(sample['wav_path'])

        if sample['avi_path'] is not None:
            video_frame = self.extractor.process_video(
                sample['avi_path'],
                start_time=sample['start_time'],
                end_time=sample['end_time']
            )
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