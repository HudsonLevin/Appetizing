import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split

# ==================== ⚡ 1. โหลดข้อมูลจาก CSV ====================
print("📌 กำลังโหลดข้อมูลจาก CSV...")

CSV_PATH = "data_from_questionaire.csv"
IMAGE_FOLDER = "datasets/"

df = pd.read_csv(CSV_PATH)

# เช็คขนาดข้อมูลทั้งหมด
print(f"📌 ข้อมูลทั้งหมดใน CSV: {len(df)} แถว")

# ถ้ามีข้อมูลมากไป ใช้แค่ 1000 แถว (หรือทั้งหมดถ้าน้อยกว่า)
df = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)

print(f"✅ ใช้ข้อมูลทั้งหมด {len(df)} คู่!")

# ==================== ⚡ 2. โหลดรูปภาพ ====================
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0  # Normalize เป็น 0-1
    return img

x1, x2, y = [], [], []

for _, row in df.iterrows():
    img1_path = os.path.join(IMAGE_FOLDER, row["Menu"], row["Image 1"])
    img2_path = os.path.join(IMAGE_FOLDER, row["Menu"], row["Image 2"])
    
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        x1.append(load_and_preprocess_image(img1_path))
        x2.append(load_and_preprocess_image(img2_path))
        y.append(row["Winner"] - 1)  # แปลง 1 -> 0, 2 -> 1

x1, x2, y = np.array(x1), np.array(x2), np.array(y)

print(f"✅ โหลดภาพ {len(x1)} คู่สำเร็จ!")

# ==================== ⚡ 3. แบ่งข้อมูล Train / Validation ====================
x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1, x2, y, test_size=0.2, random_state=42)

print(f"✅ แบ่งข้อมูล: Train {len(x1_train)} คู่, Validation {len(x1_val)} คู่!")

# ==================== ⚡ 4. Siamese Network Model ====================
def create_siamese_network():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze weights

    def create_branch():
        input_layer = Input(shape=(224, 224, 3))
        x = base_model(input_layer, training=False)
        x = GlobalAveragePooling2D()(x)
        return Model(input_layer, x)

    branch = create_branch()

    input_a = Input(shape=(224, 224, 3))
    input_b = Input(shape=(224, 224, 3))

    vector_a = branch(input_a)
    vector_b = branch(input_b)

    merged = Concatenate()([vector_a, vector_b])
    x = Dense(128, activation="relu")(merged)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[input_a, input_b], outputs=output)
    
    return model

print("🚀 กำลังสร้างโมเดล...")
model = create_siamese_network()
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])
print("✅ โมเดลถูกสร้างเรียบร้อย!")

# ==================== ⚡ 5. Data Augmentation ====================
train_datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

val_datagen = ImageDataGenerator()

def create_tf_dataset(x1, x2, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(((x1, x2), y))
    dataset = dataset.shuffle(buffer_size=len(y))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = create_tf_dataset(x1_train, x2_train, y_train)
val_dataset = create_tf_dataset(x1_val, x2_val, y_val)

# ==================== ⚡ 6. Callbacks ====================
lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
early_stopper = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# ==================== ⚡ 7. Train the Model ====================
print("🚀 เริ่มเทรนโมเดล...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[lr_reducer, early_stopper]
)

print("✅ เทรนโมเดลสำเร็จ!")

# ==================== ⚡ 8. Save the Model ====================
model.save("apetizing_model.keras")
print("💾 โมเดลถูกบันทึกเป็น 'apetizing_model.keras'")
