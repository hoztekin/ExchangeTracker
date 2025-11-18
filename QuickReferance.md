# 📋 Proje Yapısı Hızlı Referans

## 🎯 Ana Prensip

**"Ana dizin temiz kalmalı!"**

✅ Ana dizinde sadece kullanıcının sık çalıştıracağı dosyalar
❌ Test, utility ve yardımcı dosyalar alt klasörlerde

---

## 📂 Klasör Sorumlulukları

### Ana Dizin (Root)
**Sadece entry point'ler**

```
main.py              → Veri toplama
run_eda.py           → EDA çalıştırma  
app.py               → Streamlit uygulaması
setup_project.py     → İlk kurulum
```

### scripts/ Klasörü
**Kullanıcı scriptleri**

```
train_models.py              → Model eğitimi
run_technical_analysis.py    → Teknik analiz çalıştırma
backtest.py                  → Backtesting simülasyonu
```

### tests/ Klasörü
**Tüm test dosyaları**

```
test_models.py               → Model testleri
test_data_collector.py       → Veri toplama testleri
test_indicators.py           → Teknik gösterge testleri
test_integration.py          → Entegrasyon testleri
```

### src/ Klasörü
**Kütüphane kodları**

```
data/collector.py            → Veri toplama sınıfı
analysis/eda.py              → EDA sınıfı
analysis/technical.py        → Teknik analiz sınıfı
models/classifier.py         → Sınıflandırma modeli
models/regressor.py          → Regresyon modeli
models/trainer.py            → Model eğitim motoru
utils/visualization.py       → Görselleştirme araçları
utils/indicators.py          → Teknik göstergeler
```

---

## 🚀 Hızlı Kullanım

### Günlük İşlemler

```bash
# Veri güncelleme
python main.py

# EDA grafikleri
python run_eda.py

# Web uygulaması
streamlit run app.py
```

### İleri Seviye

```bash
# Model eğitimi
python scripts/train_models.py --ticker THYAO.IS

# Teknik analiz
python scripts/run_technical_analysis.py --all

# Backtesting
python scripts/backtest.py --strategy momentum

# Testler
pytest tests/
```

---

## 🔧 Proje Düzenleme

### Mevcut Yapıyı Kontrol Et

```bash
python organize_project.py
```

Bu script:
1. Mevcut dosyaları tarar
2. Taşınması gerekenleri gösterir
3. Onay alır
4. Dosyaları uygun klasörlere taşır

### Manuel Taşıma

```bash
# Test dosyaları
mv test_model.py tests/test_models.py
mv test_*.py tests/

# Script dosyaları
mv train_model.py scripts/train_models.py
mv backtest.py scripts/
```

---

## 📥 Import Yolları

### Ana dosyalardan (main.py, run_eda.py, app.py)

```python
from src.data.collector import StockDataCollector
from src.analysis.eda import ExploratoryDataAnalysis
from src.models.classifier import SignalClassifier
from src.utils.visualization import ChartGenerator
```

### Scripts'ten (scripts/*.py)

```python
import sys
sys.path.append('.')  # Ana dizini ekle

from src.models.trainer import ModelTrainer
from src.utils.helpers import load_config
```

### Tests'ten (tests/*.py)

```python
import sys
import pytest
sys.path.append('..')  # Üst dizin

from src.models.classifier import SignalClassifier
from src.data.collector import StockDataCollector
```

---

## 🗂️ Dosya İsimlendirme

### ✅ Doğru

```
main.py                          # Entry point
run_eda.py                       # Runner script
train_models.py                  # Çoğul
test_models.py                   # Çoğul
```

### ❌ Yanlış

```
test_model.py                    # Tekil (test_models.py olmalı)
train_model.py                   # Tekil (train_models.py olmalı)
test.py                          # Belirsiz
train.py                         # Belirsiz
```

---

## 📋 Checklist

### Ana Dizin Kontrolü

- [ ] `main.py` var
- [ ] `run_eda.py` var
- [ ] `app.py` var (veya `streamlit run` için)
- [ ] `setup_project.py` var
- [ ] `requirements.txt` var
- [ ] `README.md` güncel
- [ ] Test dosyaları YOK (tests/ klasöründe)
- [ ] Train dosyaları YOK (scripts/ klasöründe)

### Klasör Kontrolü

- [ ] `data/` var (raw, processed, technical)
- [ ] `src/` modüler yapı (data, analysis, models, utils)
- [ ] `scripts/` kullanıcı scriptleri
- [ ] `tests/` test dosyaları
- [ ] `outputs/` çıktı dosyaları
- [ ] `notebooks/` (opsiyonel)
- [ ] `streamlit_app/` (opsiyonel, modüler app için)

---

## 🎨 Örnek Proje Yapısı (Minimal)

```
borsa-trend-analizi/
│
├── main.py                    ✅ Entry point
├── run_eda.py                 ✅ Entry point  
├── app.py                     ✅ Entry point
├── requirements.txt           ✅ Config
├── README.md                  ✅ Docs
│
├── data/                      📁 Veri
│   ├── raw/
│   └── processed/
│
├── src/                       📁 Kod kütüphanesi
│   ├── data/
│   ├── analysis/
│   ├── models/
│   └── utils/
│
├── scripts/                   📁 Kullanıcı scriptleri
│   ├── train_models.py
│   └── backtest.py
│
├── tests/                     📁 Testler
│   └── test_models.py
│
└── outputs/                   📁 Çıktılar
    ├── eda_charts/
    └── models/
```

---

## 💡 İpuçları

1. **Ana dizin minimal tutun**
   - Sadece 4-6 Python dosyası
   - Kullanıcının ne yapacağı belli olmalı

2. **Her şey kategorize edilmeli**
   - Test → tests/
   - Script → scripts/
   - Kütüphane → src/

3. **İsimlendirme tutarlı olsun**
   - Çoğul kullan: `train_models.py`, `test_models.py`
   - Açıklayıcı: `run_technical_analysis.py` > `run_tech.py`

4. **README güncel tutun**
   - Yeni script ekleyince dokümante et
   - Kullanım örnekleri ekle

---

## ⚠️ Sık Yapılan Hatalar

### ❌ Hata 1: Ana dizin karışık

```
borsa-trend-analizi/
├── main.py
├── test1.py              ← YANLIŞ
├── test2.py              ← YANLIŞ
├── train.py              ← YANLIŞ
├── helper.py             ← YANLIŞ
└── utils.py              ← YANLIŞ
```

### ✅ Düzeltilmiş:

```
borsa-trend-analizi/
├── main.py               ← Ana entry point
├── scripts/
│   └── train_models.py   ← Taşındı
├── tests/
│   ├── test1.py          ← Taşındı
│   └── test2.py          ← Taşındı
└── src/
    └── utils/
        └── helpers.py    ← Taşındı
```

---

## 📞 Yardım

Sorun mu yaşıyorsun?

```bash
# Proje yapısını kontrol et
python organize_project.py

# Veya manuel düzenle
ls -la *.py                 # Ana dizindeki dosyaları gör
mkdir -p scripts tests      # Klasörleri oluştur
mv test_*.py tests/         # Test dosyalarını taşı
mv train_*.py scripts/      # Script dosyalarını taşı
```

---

## ✅ Son Checklist

Proje düzenini tamamladın mı?

- [ ] Ana dizinde sadece 4-6 entry point var
- [ ] Test dosyaları tests/ klasöründe
- [ ] Script dosyaları scripts/ klasöründe
- [ ] Kütüphane kodları src/ klasöründe
- [ ] Import yolları çalışıyor
- [ ] README güncel
- [ ] `pytest tests/` çalışıyor
- [ ] `python main.py` çalışıyor

---

**🎉 Proje yapın artık profesyonel ve temiz!**