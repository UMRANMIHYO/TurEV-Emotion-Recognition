# 🎙️ TurEV: Çok Modlu Duygu Tanıma Sistemi (Multimodal Emotion Recognition)

Bu proje, **Forencrypt Bilişim** yazılım mühendisliği stajı kapsamında **Umran Mihyo** tarafından geliştirilmiş yapay zeka tabanlı bir duygu tanıma (Emotion Recognition) sistemidir. 

Proje, konuşma seslerinden ve metin transkripsiyonlarından "Kızgın, Sakin, Mutlu, Üzgün" duygu durumlarını tespit etmek için **Özellik Seviyesinde Birleştirme (Feature-Level Fusion)** mimarisini kullanır.

## 🚀 Projenin Özellikleri
- **Çok Modlu Yapay Zeka:** Akustik özellikler (Wav2Vec2) ve anlamsal bağlam (BERTurk) birleştirilerek %81 doğruluk oranına ulaşılmıştır.
- **Canlı Metin Çıkarımı (ASR):** OpenAI Whisper modeli kullanılarak ses anında metne dönüştürülür.
- **İnteraktif Dashboard:** Streamlit ve Plotly kullanılarak dinamik duygu radarı ve canlı ses sinyali (waveform) arayüzü geliştirilmiştir.
- **Mikrofon & Dosya Desteği:** Kullanıcılar anında ses kaydedebilir veya .wav dosyası yükleyebilir.

## 🛠️ Kullanılan Teknolojiler
- **Derin Öğrenme Modelleri:** PyTorch, HuggingFace Transformers, Whisper, Wav2Vec2, BERTurk
- **Ses İşleme:** Librosa, Soundfile
- **Arayüz & Veri Görselleştirme:** Streamlit, Plotly, Pandas, Numpy

## 💻 Kurulum ve Çalıştırma
Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. Depoyu klonlayın ve dizine gidin.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt

  