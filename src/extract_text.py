import pandas as pd
import torch
import soundfile as sf
from transformers import pipeline
from tqdm import tqdm

def extract_transcriptions(csv_path="data/processed/metadata.csv", output_csv="data/processed/multimodal_metadata.csv"):
    """
    قراءة الملفات الصوتية المعالجة واستخراج النصوص باستخدام Whisper-small
    """
    print("جاري تحميل نموذج Whisper-small... (قد يستغرق بعض الوقت في المرة الأولى)")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-small", 
        device=device
    )
    
    df = pd.read_csv(csv_path)
    transcriptions = []
    
    print(f"بدء استخراج النصوص لـ {len(df)} ملف باستخدام {device}...")
    
    for audio_path in tqdm(df['processed_path']):
        # قراءة الملف الصوتي برمجياً لتجاوز مشكلة ffmpeg
        audio_array, sampling_rate = sf.read(audio_path)
        
        # تمرير المصفوفة الصوتية مباشرة للنموذج
        result = pipe(
            {"array": audio_array, "sampling_rate": sampling_rate}, 
            generate_kwargs={"language": "turkish"}
        )
        transcriptions.append(result["text"].strip())
        
    df['transcription'] = transcriptions
    df.to_csv(output_csv, index=False)
    print(f"\nتم الانتهاء بنجاح! تم حفظ البيانات مع النصوص في: {output_csv}")

if __name__ == "__main__":
    extract_transcriptions()