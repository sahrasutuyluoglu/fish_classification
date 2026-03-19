import os
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ------------------------
# CONFIG DOSYASINI OKU
# ------------------------
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = cfg["data"].get("train_dir", "data/Fish_Dataset")
IMG_SIZE = tuple(cfg["data"].get("image_size", [224, 224]))
BATCH_SIZE = cfg["data"].get("batch_size", 32)
EPOCHS = cfg["model"].get("epochs", 10)
LEARNING_RATE = cfg["model"].get("learning_rate", 0.001)
MODEL_SAVE_PATH = cfg["model"].get("model_save_path", "models/fish_cnn.h5")

# ------------------------
# VERİ HAZIRLAMA
# ------------------------
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ------------------------
# MODEL TANIMI
# ------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=IMG_SIZE + (3,)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------
# MODEL EĞİTİMİ
# ------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ------------------------
# MODEL KAYDETME
# ------------------------
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")