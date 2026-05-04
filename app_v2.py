import os
import warnings
# --- THE ULTIMATE CRASH PREVENTERS ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import streamlit as st
import torch
import torch.nn.functional as F
import logging
from PIL import Image
import torchvision.transforms as T

# Silence huggingface warnings in terminal
logging.getLogger("transformers").setLevel(logging.ERROR)

from src.models.prism_model_v1 import PrismMasterModel
from src.dataset.feature_extractor import MultimodalFeatureExtractor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="PrismAI V2", page_icon="🧠", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-title {
        font-size: 48px;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF904B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .classification-box {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 6px solid #FF4B4B;
        padding: 15px 20px;
        border-radius: 5px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PrismMasterModel(num_classes=4, num_datasets=2)
    
    # --- LOAD THE NEW V2 WEIGHTED MODEL ---
    model.load_state_dict(torch.load("weights/best_prism_model_v2_weighted.pth", map_location=device))
    
    model.to(device)
    model.eval()
    extractor = MultimodalFeatureExtractor()
    return model, extractor, device

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103130.png", width=80) 
    st.header("System Status")
    try:
        model, extractor, device = load_system()
        st.success("🟢 Neural Engine Online")
        system_ready = True
    except Exception as e:
        st.error("🔴 System Offline")
        system_ready = False

# --- MAIN UI ---
# --- UPDATED TITLE FOR V2 ---
st.markdown('<div class="main-title">PrismAI Engine (V2 Weighted)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="classification-box">
    <strong>🧠 Target Classifications:</strong> Neutral | Happy/Joy | Sadness | Anger
</div>
""", unsafe_allow_html=True)

input_col, output_col = st.columns([1.2, 1])

# --- INPUT SECTION ---
with input_col:
    st.subheader("Data Input")
    
    text_input = st.text_area("📝 Dialogue Transcript [Optional]", height=100)
    
    audio_tab1, audio_tab2 = st.tabs(["🎤 Record Live", "🎵 Upload File"])
    with audio_tab1:
        recorded_audio = st.audio_input("Record a voice clip directly")
    with audio_tab2:
        uploaded_audio = st.file_uploader("Upload Separate Audio (.wav)", type=["wav"])
        
    active_audio = recorded_audio if recorded_audio is not None else uploaded_audio
    
    video_tab1, video_tab2 = st.tabs(["📸 Take a Photo", "🎬 Upload Video"])
    with video_tab1:
        captured_photo = st.camera_input("Capture your facial expression")
    with video_tab2:
        uploaded_video = st.file_uploader("Upload Video (.mp4)", type=["mp4"])
    
    analyze_btn = st.button("🚀 Run Trimodal Analysis", type="primary", use_container_width=True, 
                            disabled=not (uploaded_video or captured_photo or active_audio or text_input) or not system_ready)

# --- PROCESSING & OUTPUT SECTION ---
with output_col:
    st.subheader("Analysis Results")
    
    if analyze_btn:
        try:
            with st.spinner("Processing modalities and running neural networks..."):
                
                # 1. TEXT
                if text_input.strip():
                    text_feat = extractor.process_text(text_input)
                    input_ids = text_feat['input_ids'].unsqueeze(0).to(device)
                    attention_mask = text_feat['attention_mask'].unsqueeze(0).to(device)
                else:
                    input_ids = torch.zeros((1, 128), dtype=torch.long).to(device)
                    attention_mask = torch.zeros((1, 128), dtype=torch.long).to(device)
                
                # 2. VISUAL
                video_path = None
                if captured_photo:
                    img = Image.open(captured_photo).convert('RGB')
                    transform = T.Compose([
                        T.Resize((224, 224)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    video_tensor = transform(img).unsqueeze(0).to(device)
                elif uploaded_video:
                    video_path = "temp_video.mp4"
                    with open(video_path, "wb") as f:
                        f.write(uploaded_video.read())
                    video_feat = extractor.process_video(video_path)
                    video_tensor = video_feat.unsqueeze(0).to(device)
                else:
                    video_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32).to(device)

                # 3. AUDIO
                if active_audio:
                    audio_path = "temp_audio.wav"
                    with open(audio_path, "wb") as f:
                        f.write(active_audio.read())
                    audio_feat = extractor.process_audio(audio_path)
                    os.remove(audio_path)
                elif uploaded_video and not captured_photo:
                    audio_feat = extractor.process_audio(video_path)
                else:
                    audio_feat = torch.zeros((1, 64, 250), dtype=torch.float32)
                    
                audio_tensor = audio_feat.unsqueeze(0).to(device)
                
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)
                
                dataset_id = torch.tensor([1]).to(device)
                
                # 4. INFERENCE
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask, audio_tensor, video_tensor, dataset_id)
                    logits = outputs[0]
                    
                    probabilities = F.softmax(logits, dim=1).squeeze().tolist()
                    prediction_idx = torch.argmax(logits, dim=1).item()
                    
                # 5. RESULTS DISPLAY
                emotions = ["Neutral", "Happy/Joy", "Sadness", "Anger"]
                predicted_emotion = emotions[prediction_idx]
                confidence = probabilities[prediction_idx] * 100
                
                st.success(f"**Primary Emotion Detected:** {predicted_emotion} ({confidence:.1f}%)")
                
                st.markdown("### Emotion Probability Matrix")
                for i, emotion in enumerate(emotions):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.write(f"**{emotion}**")
                    with col2:
                        st.progress(probabilities[i], text=f"{probabilities[i]*100:.1f}%")
                
        except Exception as e:
            st.error(f"⚠️ Engine Crash Intercepted: {str(e)}")