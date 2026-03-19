import os
import yaml
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------
# CONFIG DOSYASINI OKU
# ------------------------
# BASE_DIR’i proje kökü yapıyoruz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(BASE_DIR, "src", "fish_classifier", "config.yaml"), "r") as f:
    cfg = yaml.safe_load(f)

# DATA ve MODEL YOLLARI
DATA_DIR = os.path.join(BASE_DIR, cfg["data"]["train_dir"])
IMG_SIZE = tuple(cfg["data"].get("image_size", [224,224]))
BATCH_SIZE = cfg["data"].get("batch_size", 32)
MODEL_PATH = os.path.join(BASE_DIR, "models", "fish_cnn.h5")  # model proje köküne göre

# ------------------------
# VERİ HAZIRLAMA
# ------------------------
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# ------------------------
# MODEL YÜKLEME
# ------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------
# MODEL DEĞERLENDİRME
# ------------------------
loss, acc = model.evaluate(val_generator, verbose=1)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {acc*100:.2f}%")