import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np

def load_turev_metadata(data_dir):
    """قراءة مسارات ملفات TurEV وتصنيفها"""
    filepaths = []
    labels = []
    emotions = ['angry', 'calm', 'happy', 'sad']
    
    for emotion in emotions:
        folder_path = os.path.join(data_dir, emotion)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".wav"):
                    filepaths.append(os.path.join(folder_path, filename))
                    labels.append(emotion)
                    
    return pd.DataFrame({'file_path': filepaths, 'emotion': labels})

def preprocess_audio(file_path, output_path, target_sr=16000, top_db=30):
    """تطبيق المعالجة المسبقة على ملف صوتي واحد"""
    # 1. قراءة الملف وتغيير التردد إلى 16000
    y, sr = librosa.load(file_path, sr=target_sr)
    
    # 2. قص فترات الصمت
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    
    # 3. تسوية السعة الصوتية
    y_normalized = librosa.util.normalize(y_trimmed)
    
    # التأكد من وجود مجلد الحفظ (سيقوم بإنشاء مجلد المشاعر إذا لم يكن موجوداً)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # حفظ الملف المعالج
    sf.write(output_path, y_normalized, target_sr)
    return output_path

def process_all_files(metadata_df, processed_dir="data/processed"):
    """تطبيق المعالجة على جميع الملفات وحفظها"""
    processed_paths = []
    print("جاري معالجة الملفات الصوتية وقص الصمت... يرجى الانتظار (قد تستغرق العملية بضع دقائق).")
    
    for index, row in metadata_df.iterrows():
        emotion = row['emotion']
        filename = os.path.basename(row['file_path'])
        
        # بناء مسار الحفظ الجديد (مثال: data/processed/angry/file.wav)
        output_path = os.path.join(processed_dir, emotion, filename)
        
        # معالجة وحفظ
        preprocess_audio(row['file_path'], output_path)
        processed_paths.append(output_path)
        
        # طباعة مؤشر التقدم
        if (index + 1) % 100 == 0:
            print(f"تمت معالجة {index + 1} ملف...")
            
    # تحديث البيانات الوصفية بالمسارات الجديدة
    metadata_df['processed_path'] = processed_paths
    
    # حفظ البيانات في ملف CSV لسهولة الوصول إليها في الخطوات القادمة
    csv_path = os.path.join(processed_dir, "metadata.csv")
    metadata_df.to_csv(csv_path, index=False)
    
    print("\nانتهت المعالجة بنجاح!")
    print(f"تم حفظ الملفات المعالجة في: {processed_dir}")
    print(f"تم حفظ خريطة البيانات في: {csv_path}")
    return metadata_df

if __name__ == "__main__":
    raw_data_path = "data/raw"
    
    # 1. استخراج المسارات الأصلية
    metadata_df = load_turev_metadata(raw_data_path)
    
    # 2. بدء المعالجة
    if len(metadata_df) > 0:
        process_all_files(metadata_df)
    else:
        print("لم يتم العثور على ملفات صوتية، يرجى التأكد من مسار مجلد data/raw.")