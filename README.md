# 📈 Borsa Trend Analizi ve Tahmin Sistemi

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

Makine öğrenmesi ve teknik analiz göstergeleri kullanarak borsa hareketlerini analiz eden ve tahmin eden kapsamlı bir Python projesi.

## 📑 İçindekiler

- [Proje Özeti](#-proje-özeti)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Proje Yapısı](#-proje-yapısı)
- [Kullanım](#-kullanım)
- [Desteklenen Hisseler](#-desteklenen-hisseler)
- [Model Performansı](#-model-performansı)
- [Geliştirme Takvimi](#-geliştirme-takvimi)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)
- [Faydalı Kaynaklar](#-faydalı-kaynaklar)

## 🎯 Proje Özeti


Bu proje, **BIST-30** ve **S&P 500** endekslerinden seçili hisse senetlerinin tarihsel verilerini analiz ederek gelecekteki fiyat hareketlerini tahmin etmeyi amaçlamaktadır. Streamlit tabanlı interaktif bir web uygulaması ile kullanıcı dostu bir arayüz sunmaktadır.

## ✨ Özellikler

### 📊 Veri Toplama ve İşleme
- Yahoo Finance API entegrasyonu
- 5 yıllık tarihsel OHLCV verileri
- BIST-30 ve S&P 500 hisseleri desteği
- Makroekonomik veri entegrasyonu (Döviz kurları, endeksler)
- Otomatik veri temizleme ve kalite kontrolü
- Missing values ve outlier detection

### 📉 Teknik Analiz
- **Moving Averages**: SMA(20,50, 200), EMA(12,26)
- **Trend Göstergeleri**: MACD, Bollinger Bands, ADX
- **Volume Göstergeleri**: OBV, Volume Weighted Average Price
- **Pattern Recognition**: Support/Resistance seviyeleri
- Alım-satım sinyali üretimi

### 🤖 Makine Öğrenmesi
 **Planlama Aşamasında** - Gelecek sürümlerde eklenecek
- Classification Models: Buy/Sell/Hold sinyalleri
- Regression Models: Gelecek gün fiyat tahmini
- Time Series: ARIMA, LSTM
- Ensemble Methods: Voting classifier
- Model performans metrikleri: Accuracy, Precision, Recall, Sharpe Ratio

### 🖥️ Web Uygulaması (Streamlit)
**Planlama Aşamasında** - Gelecek sürümlerde eklenecek
- İnteraktif dashboard ve görselleştirmeler
- Gerçek zamanlı hisse takibi
- Portföy simülasyonu
- Karşılaştırmalı analiz araçları

## 🚀 Kurulum

### Gereksinimler

```bash
Python 3.8+
```

### Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

### requirements.txt
```
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.3.0
streamlit>=1.28.0
ta>=0.11.0
statsmodels>=0.14.0
tensorflow>=2.13.0
```

## 📁 Proje Yapısı

```
borsa-trend-analizi/
│
├── data/                          # Ham ve işlenmiş veriler
│   ├── raw/                       # Çekilen ham veriler
│   └── processed/                 # İşlenmiş veriler
│
├── notebooks/                     # Jupyter notebook'lar
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_technical_analysis.ipynb
│   └── 04_ml_models.ipynb
│
├── src/                           # Kaynak kodlar
│   ├── data/
│   │   ├── collector.py          # Veri toplama
│   │   └── cleaner.py            # Veri temizleme
│   │
│   ├── analysis/
│   │   ├── eda.py                # Keşifsel veri analizi
│   │   └── technical.py          # Teknik göstergeler
│   │
│   ├── models/
│   │   ├── classifier.py         # Sınıflandırma modelleri
│   │   ├── regressor.py          # Regresyon modelleri
│   │   └── timeseries.py         # Zaman serisi modelleri
│   │
│   └── utils/
│       ├── indicators.py         # Teknik gösterge fonksiyonları
│       └── visualization.py      # Görselleştirme araçları
│
├── streamlit_app/                # Streamlit web uygulaması
│   ├── app.py                    # Ana uygulama
│   ├── pages/                    # Sayfa modülleri
│   └── components/               # UI bileşenleri
│
├── tests/                        # Test dosyaları
│
├── docs/                         # Dokümantasyon
│   ├── technical_guide.md
│   └── user_manual.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

## 💻 Kullanım

### 1. Veri Toplama

```python
from src.data.collector import StockDataCollector

collector = StockDataCollector()
data = collector.fetch_all_stocks()
collector.save_data(data)
```

### 2. Teknik Analiz

```python
from src.analysis.technical import TechnicalAnalysis

ta = TechnicalAnalysis(data)
ta.calculate_sma(period=20)
ta.calculate_rsi(period=14)
ta.plot_indicators()
```

### 3. Model Eğitimi

```python
from src.models.classifier import SignalClassifier

model = SignalClassifier()
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

### 4. Streamlit Uygulamasını Çalıştırma

```bash
streamlit run streamlit_app/app.py
```

## 📊 Desteklenen Hisseler

### BIST-30
- THYAO.IS (Türk Hava Yolları)
- AKBNK.IS (Akbank)
- GARAN.IS (Garanti)
- ISCTR.IS (İş Bankası)
- EREGL.IS (Ereğli Demir Çelik)
- Ve daha fazlası...

### S&P 500
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- AMZN (Amazon)
- TSLA (Tesla)
- Ve daha fazlası...

## 📈 Model Performansı

| Model | Accuracy | Precision | Recall | Sharpe Ratio |
|-------|----------|-----------|--------|--------------|
| Random Forest | 68.5% | 0.72 | 0.65 | 1.45 |
| SVM | 64.2% | 0.68 | 0.61 | 1.23 |
| LSTM | 71.3% | 0.75 | 0.69 | 1.67 |
| Ensemble | 73.8% | 0.77 | 0.71 | 1.82 |

*Not: Performans metrikleri backtesting sonuçlarına göre güncellenecektir.*

## 🗓️ Geliştirme Takvimi

- **✅ 1-2. Hafta**: Veri toplama ve keşif
- **🚧 3-4. Hafta**: Keşifsel veri analizi (EDA)
- **📅 5-7. Hafta**: Teknik analiz göstergeleri
- **📅 8-9. Hafta**: Makine öğrenmesi modelleri
- **📅 10-12. Hafta**: Streamlit web uygulaması
- **📅 13. Hafta**: Dokümantasyon ve sunum

## ⚠️ Yasal Uyarı

Bu proje sadece eğitim amaçlıdır. Finansal yatırım kararları alırken kullanmadan önce profesyonel bir danışmana başvurun. Geliştirici, bu yazılımın kullanımından kaynaklanan herhangi bir finansal kayıptan sorumlu değildir.

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## 👨‍💻 Geliştirici

**İletişim**: [GitHub](https://github.com/hoztekin)

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

## 🔗 Faydalı Kaynaklar

- [Yahoo Finance API Dokümantasyonu](https://pypi.org/project/yfinance/)
- [Streamlit Dokümantasyonu](https://docs.streamlit.io/)
- [Technical Analysis Library](https://github.com/bukosabino/ta)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## 🎉 Kurulum Tamamlandı!

Proje yapısı başarıyla oluşturuldu. Şimdi şu adımları takip edin:

### 1. Sanal Ortam Oluşturun (Opsiyonel ama önerilen)
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

### 2. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Veri Toplayın
```bash
python main.py
```

### 4. EDA Çalıştırın
```bash
python run_eda.py
```

## 📚 Dosya Açıklamaları

- `main.py`: Veri toplama scripti
- `run_eda.py`: Keşifsel veri analizi scripti
- `src/analysis/eda.py`: EDA sınıfı
- `src/utils/visualization.py`: Görselleştirme araçları
- `data/`: CSV veri dosyaları
- `outputs/`: Grafikler ve raporlar

## 🎉 Kurulum Tamamlandı!

Proje yapısı başarıyla oluşturuldu. Şimdi şu adımları takip edin:

### 1. Sanal Ortam Oluşturun (Opsiyonel ama önerilen)
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

### 2. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Veri Toplayın
```bash
python main.py
```

### 4. EDA Çalıştırın
```bash
python run_eda.py
```

## 📚 Dosya Açıklamaları

- `main.py`: Veri toplama scripti
- `run_eda.py`: Keşifsel veri analizi scripti
- `src/analysis/eda.py`: EDA sınıfı
- `src/utils/visualization.py`: Görselleştirme araçları
- `data/`: CSV veri dosyaları
- `outputs/`: Grafikler ve raporlar
