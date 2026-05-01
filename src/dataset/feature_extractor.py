import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import soundfile as sf
from transformers import RobertaTokenizer
from src.models.video_encoder import VideoEncoder

class MultimodalFeatureExtractor:
    def __init__(self, text_model_name='roberta-base', max_text_len=128, max_audio_len=250, sample_rate=16000):
        self.tokenizer = RobertaTokenizer.from_pretrained(text_model_name)
        self.max_text_len = max_text_len
        self.max_audio_len = max_audio_len 
        self.sample_rate = sample_rate
        
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.video_encoder = VideoEncoder()

    def process_text(self, text):
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

    def process_audio(self, filepath):
        try:
            data, sr = sf.read(filepath)
            waveform = torch.FloatTensor(data)
            
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
                
            if sr != self.sample_rate:
                resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
                
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            mel_spec = self.mel_spectrogram(waveform)
            log_mel_spec = self.amplitude_to_db(mel_spec)
            
            # --- NEW: Pad or Truncate Audio to max_audio_len ---
            time_steps = log_mel_spec.shape[-1]
            if time_steps < self.max_audio_len:
                # Pad with zeros on the time dimension
                padding = self.max_audio_len - time_steps
                log_mel_spec = F.pad(log_mel_spec, (0, padding))
            elif time_steps > self.max_audio_len:
                # Truncate to max length
                log_mel_spec = log_mel_spec[:, :, :self.max_audio_len]
                
            return log_mel_spec
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros((1, 64, self.max_audio_len))

    def process_video(self, video_path, start_time=None, end_time=None):
        return self.video_encoder.extract_frame(video_path, start_time, end_time)

if __name__ == "__main__":
    extractor = MultimodalFeatureExtractor()
    text_features = extractor.process_text("Excuse me.")
    print(f"Text Tokens Shape: {text_features['input_ids'].shape}")
    print("Feature Extractor initialized successfully.")