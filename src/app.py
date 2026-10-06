import streamlit as st
import librosa
import torch
import torch.nn as nn
import numpy as np
import io
import plotly.express as px
import pandas as pd
import torch.nn.functional as F
from transformers import pipeline, AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor

# ==========================================
# 1. تعريف معمارية النموذج المدمج (من تدريبك)
# ==========================================
class MultimodalEmotionClassifier(nn.Module):
    def __init__(self, text_model_path, audio_model_path, num_labels=4):
        super(MultimodalEmotionClassifier, self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_path)
        self.audio_model = AutoModel.from_pretrained(audio_model_path)
        
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.audio_model.parameters():
            param.requires_grad = False
            
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768 + 768, num_labels)
        
    def forward(self, input_ids, attention_mask, input_values, labels=None):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :] 
        
        audio_outputs = self.audio_model(input_values=input_values)
        audio_features = torch.mean(audio_outputs.last_hidden_state, dim=1) 
        
        combined_features = torch.cat((text_features, audio_features), dim=1)
        
        x = self.dropout(combined_features)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            
        return {"loss": loss, "logits": logits}

# ==========================================
# 2. تحميل النماذج والأوزان في الذاكرة المؤقتة
# ==========================================
@st.cache_resource
def load_models():
    asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")
    
    text_model_name = "dbmdz/bert-base-turkish-cased"
    audio_model_name = "facebook/wav2vec2-base"
    
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
    
    emotion_model = MultimodalEmotionClassifier(text_model_name, audio_model_name, num_labels=4)
    
    weights_path = "models/turev_multimodal_final/multimodal_weights.pth"
    emotion_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')), strict=False)
    emotion_model.eval() 
    
    return asr_model, tokenizer, feature_extractor, emotion_model

asr_pipeline, tokenizer, feature_extractor, emotion_model = load_models()

# ==========================================
# 3. إعدادات الواجهة ودوال المساعدة
# ==========================================
# تغيير التخطيط ليكون عريضاً (wide) ليناسب الداشبورد الاحترافي
st.set_page_config(page_title="TurEV Emotion Recognition", page_icon="🎙️", layout="wide")

def reset_audio():
    if "file_uploader_key" in st.session_state:
        del st.session_state["file_uploader_key"]
    if "audio_recorder_key" in st.session_state:
        del st.session_state["audio_recorder_key"]

emotion_labels = {
    0: "😡 Kızgın (Angry)",
    1: "😌 Sakin (Calm)",
    2: "😊 Mutlu (Happy)",
    3: "😢 Üzgün (Sad)"
}

st.title("🎙️ Çok Modlu Duygu Tanıma Sistemi")
st.markdown("**Forencrypt Bilişim - Staj Projesi** | Geliştirici: Umran Mihyo")
st.divider()

st.header("1. Ses Girişi")

input_method = st.radio("Giriş yöntemini seçin:", ["📁 Dosya Yükle (Upload)", "🎤 Ses Kaydet (Record)"], horizontal=True)

audio_file = None

if input_method == "📁 Dosya Yükle (Upload)":
    uploaded_file = st.file_uploader("Lütfen analiz edilecek ses dosyasını seçin (.wav, .mp3)", type=["wav", "mp3"], key="file_uploader_key")
    if uploaded_file is not None:
        audio_file = uploaded_file

elif input_method == "🎤 Ses Kaydet (Record)":
    recorded_file = st.audio_input("Mikrofonunuzu kullanarak ses kaydedin", key="audio_recorder_key")
    if recorded_file is not None:
        audio_file = recorded_file

# ==========================================
# 4. المعالجة والتحليل (Inference)
# ==========================================
if audio_file is not None:
    st.divider()
    st.header("2. Duygu Analizi")
    
    st.audio(audio_file)
    
    # أزرار التحكم في الأعلى
    col_btn1, col_btn2 = st.columns([4, 1])
    
    with col_btn1:
        analyze_button = st.button("🧠 Duyguyu Analiz Et", use_container_width=True)
    with col_btn2:
        st.button("🔄 Sıfırla (Temizle)", on_click=reset_audio, use_container_width=True)

    # تنفيذ التحليل إذا تم الضغط على الزر (خارج الأعمدة الضيقة)
    if analyze_button:
        audio_bytes = audio_file.read()
        
        with st.spinner("⏳ Ses metne dönüştürülüyor (ASR)..."):
            transcription = asr_pipeline(audio_bytes)["text"]
            st.info(f"**🗣️ Konuşma Metni:** {transcription}")
            
        with st.spinner("🧠 Çok Modlu Yapay Zeka Duygu Analizi Yapıyor..."):
            try:
                audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
                target_length = 48000
                if len(audio_array) > target_length:
                    audio_array = audio_array[:target_length]
                else:
                    audio_array = np.pad(audio_array, (0, max(0, target_length - len(audio_array))), "constant")
                
                audio_inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
                input_values = audio_inputs.input_values
                
                text_inputs = tokenizer(transcription, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
                input_ids = text_inputs["input_ids"]
                attention_mask = text_inputs["attention_mask"]
                
                with torch.no_grad():
                    outputs = emotion_model(input_ids=input_ids, attention_mask=attention_mask, input_values=input_values)
                    logits = outputs["logits"]
                    
                    probs = F.softmax(logits, dim=-1)[0].numpy()
                    predicted_class_id = np.argmax(probs)
                    predicted_emotion = emotion_labels.get(predicted_class_id, "Bilinmeyen")
                
                st.success(f"✅ Analiz Tamamlandı! Tespit Edilen Duygu: **{predicted_emotion}**")
                st.markdown("---")
                
                # إنشاء أعمدة لعرض الرسوميات بمسافة واسعة ومريحة
                col_wave, col_radar = st.columns([2, 1.5])
                
                with col_wave:
                    st.markdown("### 📈 Canlı Ses Sinyali (Waveform)")
                    downsampled_audio = audio_array[::10] 
                    st.line_chart(downsampled_audio, height=350, use_container_width=True)
                    
                with col_radar:
                    st.markdown("### 🎯 Duygu Radarı")
                    df_radar = pd.DataFrame({
                        'r': probs,
                        'theta': ["Kızgın", "Sakin", "Mutlu", "Üzgün"]
                    })
                    fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True)
                    fig.update_traces(fill='toself', line_color='#8B5CF6') 
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        polar=dict(
                            radialaxis=dict(visible=False, range=[0, 1]),
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        margin=dict(l=40, r=40, t=20, b=20), # زيادة الهوامش لتجنب قطع الكلمات
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Analiz sırasında bir hata oluştu: {str(e)}")