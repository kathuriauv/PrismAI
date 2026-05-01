import os
import subprocess
import imageio_ffmpeg

def convert_meld_to_wav(video_dir):
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    mp4_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4') and not f.startswith('._')]
    print(f"Found {len(mp4_files)} videos. Starting bulletproof conversion...")
    
    for i, filename in enumerate(mp4_files):
        mp4_path = os.path.join(video_dir, filename)
        wav_path = mp4_path.replace('.mp4', '.wav')
        
        if os.path.exists(wav_path):
            continue

        cmd = [ffmpeg_exe, "-y", "-i", mp4_path, "-ac", "1", "-ar", "16000", wav_path]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            print(f"Failed on {filename}: {e}")
            
        if (i + 1) % 500 == 0:
            print(f"Converted {i + 1}/{len(mp4_files)} files...")
            
    print("Conversion completely finished!")

if __name__ == "__main__":
    
    meld_video_folder = r"C:\Users\kathu\OneDrive\Desktop\Projects\PrismAI_v1\data\raw\MELD-RAW\MELD.Raw\train\train_splits"
    convert_meld_to_wav(meld_video_folder)