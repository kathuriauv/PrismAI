import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import RobertaTokenizer

class MultimodalFeatureExtractor:
    def __init__(self, text_model_name='roberta-base', max_text_len=128, sample_rate=16000):
        """
        The Translator. Converts raw text and audio files into AI-ready math.
        """
        # We use RoBERTa 
        self.tokenizer = RobertaTokenizer.from_pretrained(text_model_name)
        self.max_text_len = max_text_len
        
        # Audio Setup 
        self.sample_rate = sample_rate
        # Log-Mel Spectrograms
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def process_text(self, text: str):
        """Turns 'I am happy' into padded RoBERTa tensor tokens."""
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

    def process_audio(self, filepath: str):
        """Loads a .wav file using soundfile to bypass broken PyTorch codec routers."""
        try:
            # 1. Load the raw data and sample rate using soundfile instead of torchaudio
            data, sr = sf.read(filepath)
            
            # 2. Convert to a torch tensor
            # soundfile returns (samples, channels) - we need (channels, samples)
            waveform = torch.FloatTensor(data)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0) # Add channel dim for mono
            else:
                waveform = waveform.T # Transpose to (channels, samples)
                
            # 3. Standardize the sample rate to 16kHz
            if sr != self.sample_rate:
                resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
                
            # 4. Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # 5. Create the Log-Mel Spectrogram
            mel_spec = self.mel_spectrogram(waveform)
            log_mel_spec = self.amplitude_to_db(mel_spec)
            
            return log_mel_spec
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return torch.zeros((1, 64, 100))
if __name__ == "__main__":
    # TEST BLOCK 
    extractor = MultimodalFeatureExtractor()
    
    # Test Text
    text_features = extractor.process_text("Excuse me.")
    print("Text Tokens Shape:", text_features['input_ids'].shape)
    
    # Test audio after editing it.
    print("Feature Extractor successfully initialized and text tokenized!")