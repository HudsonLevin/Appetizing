import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random

# ==================== ⚡ โหลดโมเดล ====================
MODEL_PATH = "apetizing_model.keras"
IMAGE_FOLDER = "test_images/"

print("📌 กำลังโหลดโมเดล...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ โหลดโมเดลสำเร็จ!")

# ==================== ⚡ เลือกภาพจากแต่ละหมวด ====================
categories = ["Burger", "Dessert", "Pizza", "Ramen", "Sushi"]
selected_images = {}

for category in categories:
    category_path = os.path.join(IMAGE_FOLDER, category)
    image_files = [f for f in os.listdir(category_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if len(image_files) < 2:
        print(f"❌ หมวด {category} มีภาพไม่พอ (ต้องมีอย่างน้อย 2 รูป)")
        continue

    selected_images[category] = random.sample(image_files, 2)

# ==================== ⚡ โหลดและแสดงผลภาพ ====================
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # Normalize เป็น 0-1
    return img, img_array

for category, (img1_name, img2_name) in selected_images.items():
    img1_path = os.path.join(IMAGE_FOLDER, category, img1_name)
    img2_path = os.path.join(IMAGE_FOLDER, category, img2_name)

    img1, img1_array = load_and_preprocess_image(img1_path)
    img2, img2_array = load_and_preprocess_image(img2_path)

    # ========== ⚡ ให้โมเดลทำนาย ==========
    prediction = model.predict([np.expand_dims(img1_array, axis=0), np.expand_dims(img2_array, axis=0)])
    pred_score = prediction[0][0]
    predicted_winner = 1 if pred_score < 0.5 else 2

    # ========== ⚡ แสดงผลภาพ ==========
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

    plt.suptitle(
        f"🍔 {category} | Model Select: Image {predicted_winner}",
        fontsize=14,
        fontweight="bold",
        color="green"
    )

    plt.show()
