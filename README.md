# Fish Classification 🐟

Derin öğrenme tabanlı balık türü sınıflandırma projesi. EfficientNetB0 transfer learning mimarisi kullanılarak 9 farklı balık türü sınıflandırılmaktadır.

## Teknik Özellikler

- **Model:** EfficientNetB0 (ImageNet ağırlıkları ile transfer learning)
- **Veri artırma:** RandomFlip, RandomRotation, RandomZoom, RandomBrightness
- **Pipeline:** `tf.data` API ile önbellekleme ve prefetch
- **Callback:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Metrikler:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Sınıflar

| # | Balık Türü |
|---|-----------|
| 1 | Black Sea Sprat |
| 2 | Gilt-Head Bream |
| 3 | Hourse Mackerel |
| 4 | Red Mullet |
| 5 | Red Sea Bream |
| 6 | Sea Bass |
| 7 | Shrimp |
| 8 | Striped Red Mullet |
| 9 | Trout |

## Proje Yapısı

```
fish_classification/
├── data/
│   └── Fish_Dataset/
│       ├── Black Sea Sprat/
│       ├── Gilt-Head Bream/
│       └── ...
├── models/
│   ├── figures/          # Eğitim grafikleri ve confusion matrix
│   └── logs/
├── scripts/
│   ├── train.py          # Eğitim scripti
│   └── evaluate.py       # Değerlendirme scripti
├── src/
│   └── fish_classifier/
│       ├── config.yaml   # Tüm hiperparametreler
│       ├── data_loader.py
│       ├── evaluate.py
│       ├── model.py
│       ├── preprocessing.py
│       ├── train.py
│       └── utils.py
├── main.py
└── requirements.txt
```

## Kurulum

```bash
# Repo klonla
git clone https://github.com/sahrasutuyluoglu/fish_classification.git
cd fish_classification

# Python 3.11 ile sanal ortam oluştur
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt
```

## Veri Seti

Kaggle'daki [A Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset) kullanılmaktadır.

İndirdikten sonra `data/Fish_Dataset/` klasörüne koy. Her sınıf ayrı bir klasörde olmalıdır.

> ⚠️ Veri seti repo'ya dahil edilmemiştir. `data/README.txt` dosyasındaki talimatları izleyerek indirebilirsiniz.

## Kullanım

**Eğitim:**
```bash
python scripts/train.py
# veya özel config ile
python scripts/train.py --config src/fish_classifier/config.yaml
```

**Değerlendirme:**
```bash
python scripts/evaluate.py
```

**Ana giriş noktası:**
```bash
python main.py --mode train
python main.py --mode evaluate
```

## Konfigürasyon

Tüm hiperparametreler `src/fish_classifier/config.yaml` üzerinden yönetilmektedir:

```yaml
model:
  architecture: efficientnetb0   # efficientnetb0 | baseline_cnn
  epochs: 20
  learning_rate: 0.001
  fine_tune: false               # true: base model üst katmanları da eğitilir
```

## Sonuçlar

> Eğitim tamamlandıktan sonra buraya eklenecek.

| Metrik | Değer |
|--------|-------|
| Validation Accuracy | - |
| Validation Loss | - |

Confusion matrix ve eğitim grafikleri `models/figures/` klasöründe saklanmaktadır.

## Gereksinimler

- Python 3.11
- TensorFlow 2.21.0
- scikit-learn 1.8.0

Tüm bağımlılıklar için `requirements.txt` dosyasına bakınız.
