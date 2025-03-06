import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1️⃣ โหลดโมเดล
model = load_model("apetizing_model.keras")
print("✅ โหลดโมเดลสำเร็จ!")

# 2️⃣ โหลดไฟล์ test.csv
CSV_PATH = "Test Set Samples/test.csv"
IMAGE_FOLDER = "Test Set Samples/Test Images"  # ตำแหน่งของรูปภาพ

df_test = pd.read_csv(CSV_PATH)

# 3️⃣ ฟังก์ชันโหลดและแปลงรูปภาพ
def load_and_preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize 0-1
    return img

x1, x2 = [], []

for _, row in df_test.iterrows():
    img1_path = f"{IMAGE_FOLDER}/{row['Image 1']}"  # หรือเปลี่ยนนามสกุลให้ตรง
    img2_path = f"{IMAGE_FOLDER}/{row['Image 2']}"

    x1.append(load_and_preprocess_image(img1_path))
    x2.append(load_and_preprocess_image(img2_path))

x1, x2 = np.array(x1), np.array(x2)

print(f"✅ โหลดรูปภาพ {len(x1)} คู่สำเร็จ!")

# 4️⃣ ทำนายผล
y_pred_prob = model.predict([x1, x2])  # ได้ค่า probability (0 - 1)
y_pred = (y_pred_prob > 0.5).astype(int)  # แปลงค่าเป็น 0 หรือ 1
df_test["Winner"] = y_pred + 1  # กลับเป็น 1 หรือ 2

# 5️⃣ บันทึกไฟล์ CSV ใหม่
df_test.to_csv("Test Set Samples/test.csv", index=False)
print("✅ บันทึกผลลัพธ์ลง test.csv สำเร็จ!")
