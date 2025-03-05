import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ==================== ⚡ โหลดโมเดล ====================
MODEL_PATH = "apetizing_model.keras"
CSV_PATH = "data_from_questionaire.csv"
IMAGE_FOLDER = "datasets/"
CATEGORIES = ["Burger", "Dessert", "Pizza", "Ramen", "Sushi"]

print("📌 กำลังโหลดโมเดล...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ โหลดโมเดลสำเร็จ!")

# ==================== ⚡ โหลดข้อมูลจาก CSV ====================
df = pd.read_csv(CSV_PATH)

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize เป็น 0-1
    return img, img_array

def plot_images(img1, img2, img1_name, img2_name, pred, true_winner):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img1)
    axes[0].axis("off")
    axes[0].set_title(f"Image 1", fontsize=14, fontweight="bold")

    axes[1].imshow(img2)
    axes[1].axis("off")
    axes[1].set_title(f"Image 2", fontsize=14, fontweight="bold")

    # ใส่ชื่อไฟล์ใต้ภาพ
    fig.text(0.25, 0.02, f"{img1_name}", ha="center", fontsize=10, color="blue")
    fig.text(0.75, 0.02, f"{img2_name}", ha="center", fontsize=10, color="blue")

    predicted_winner = 1 if pred < 0.5 else 2
    color = "green" if predicted_winner == true_winner else "red"

    plt.suptitle(
        f"✅ Model Select: Image {predicted_winner} | Real: Image {true_winner}",
        fontsize=14,
        fontweight="bold",
        color=color,
    )

    plt.show()

# ==================== ⚡ ทดสอบโมเดลกับภาพจริง ====================
for category in CATEGORIES:
    print(f"\n🔍 ทดสอบหมวด: {category}")

    # ดึงข้อมูลที่เป็นของหมวดนี้
    df_category = df[df["Menu"] == category]
    if df_category.empty:
        print(f"❌ ไม่มีข้อมูลใน CSV สำหรับหมวด {category}")
        continue

    # สุ่มเลือก 1 แถวจาก CSV
    sample = df_category.sample(1).iloc[0]
    img1_name = sample["Image 1"]
    img2_name = sample["Image 2"]
    img1_path = os.path.join(IMAGE_FOLDER, category, img1_name)
    img2_path = os.path.join(IMAGE_FOLDER, category, img2_name)
    true_winner = sample["Winner"]  # ค่าเฉลยจากคนจริงๆ

    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        print(f"❌ ไฟล์ภาพไม่ครบ: {img1_path} หรือ {img2_path} ไม่มีอยู่")
        continue

    img1, img1_array = load_and_preprocess_image(img1_path)
    img2, img2_array = load_and_preprocess_image(img2_path)

    # ==================== ⚡ ให้โมเดลทำนาย ====================
    prediction = model.predict([np.expand_dims(img1_array, axis=0), np.expand_dims(img2_array, axis=0)])
    pred_score = prediction[0][0]

    print(f"🔮 โมเดลทำนาย: {pred_score:.4f} (ใกล้ 0 → Image 1 ชนะ, ใกล้ 1 → Image 2 ชนะ)")
    print(f"🖼️ Image 1: {img1_name}  |  Image 2: {img2_name}")

    # ==================== ⚡ แสดงผลภาพ ====================
    plot_images(img1, img2, img1_name, img2_name, pred_score, true_winner)
