# 🐟 Fish Classification

## Proje Hakkında
Bu proje, farklı balık türlerini sınıflandırmak için geliştirilmiş bir **derin öğrenme tabanlı görüntü sınıflandırma** uygulamasıdır.  
Kullanıcıların kendi balık görüntülerini sınıflandırabilmesi ve modeli eğitebilmesi için bir pipeline sunar.

- Veri yükleme (`ImageDataGenerator`)
- Basit CNN modeli
- Model eğitimi ve kaydetme
- Hiperparametre yönetimi (`config.yaml`)

```text
fish_classification/
|-- data/
|   `-- Fish_Dataset/
|       |-- Black Sea Sprat/
|       |-- Gilt-Head Bream/
|       |-- Hourse Mackerel/
|       |-- Red Mullet/
|       |-- Red Sea Bream/
|       |-- Sea Bass/
|       |-- Shrimp/
|       |-- Striped Red Mullet/
|       `-- Trout/
|-- models/
|   |-- figures/
|   `-- logs/
|-- outputs/
|-- scripts/
|   |-- evaluate.py
|   `-- train.py
|-- src/
|   `-- fish_classifier/
|       |-- __init__.py
|       |-- config.yaml
|       |-- data_loader.py
|       |-- evaluate.py
|       |-- model.py
|       |-- preprocessing.py
|       |-- train.py
|       `-- utils.py
|-- .gitignore
|-- README.md
|-- main.py
`-- requirements.txt
```
## Veri Seti

- Projede kullanılan veri seti: `data/Fish_Dataset`  
- İçerik: 9 farklı balık türü (örn. Salmon, Trout, Tuna, Cod, vs.)  
- Her sınıf için ayrı klasörler:  

> ⚠️ Dataset repoda değil, ancak `data` klasöründe bir **README içinde dataset linki** bulunmaktadır. Kullanıcılar oradaki talimatları izleyerek veri setini indirebilir.

## Kurulum
```bash
# Repo klonla
git clone https://github.com/sahrasutuyluoglu/fish_classification.git
cd fish_classification

# Sanal ortam oluştur (opsiyonel)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Gerekli paketleri yükle
pip install -r requirements.txt
```
## Model Eğitimi
````
python src/fish_classifier/train.py --config config.yaml

````

- Eğitim tamamlandığında model models/fish_model.h5 olarak kaydedilir.

- Eğitim süresini ve batch boyutunu config.yaml üzerinden ayarlayabilirsiniz.

## Tahmin
``
python src/fish_classifier/predict.py --image_path path/to/image.jpg
``
Çıktı: tahmin edilen balık türü

Örnek çıktı:

Image: salmon_01.jpg
Predicted Class: Salmon
Confidence: 92.4%
## Değerlendirme

- Modelin başarımı accuracy ile ölçülür.

- İleri seviye kullanım için confusion matrix ve classification report eklenebilir.

Örnek Confusion Matrix:

## İyileştirme Önerileri

Transfer learning (ResNet, EfficientNet) kullanarak performans artırma

Veri augmentasyonu (flip, rotate, zoom, brightness)

Early stopping ve learning rate scheduler ekleme

Daha detaylı değerlendirme metrikleri (precision, recall, F1-score)
