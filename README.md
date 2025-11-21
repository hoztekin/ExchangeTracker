# 📊 ExchangeTrack - Borsa Trend Analizi ve Tahmin Sistemi

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![Status](https://img.shields.io/badge/Status-✅%20Tamamlandı-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

**Makine öğrenmesi ve teknik analiz kullanarak BIST-30 ve S&P 500 hisse senetlerini analiz eden ve tahmin eden profesyonel fintech sistemi.**

> 🎯 **13 haftalık yoğun akademik proje → Üretim ortamında çalışan sistem**

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum
```bash
# Repo klonla
git clone https://github.com/hoztekin/ExchangeTracker
cd exchangetrack

# Sanal ortam
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Paketler
pip install -r requirements.txt
```

### 2. Streamlit Dashboard Çalıştır
```bash
streamlit run app.py
```
**Tarayıcı otomatik açılır:** `http://localhost:8501`

### 3. Docker (Opsiyonel)
```bash
docker-compose up
```

---

## ✨ Temel Özellikler

### 📊 Interactive Dashboard
- ✅ Gerçek zamanlı tahminler (Yarının fiyatı)
- ✅ BUY/SELL/HOLD sinyalleri
- ✅ Teknik göstergeler (RSI, MACD, Bollinger Bands, ATR)
- ✅ Backtest performans metrikleri
- ✅ İnteraktif Plotly grafikler
- ✅ Çoklu hisse analizi (BIST-30 + S&P 500)

### 🤖 Makine Öğrenmesi
- ✅ **Regression modelleri:** Ridge, LassoLarsCV, HuberRegressor (R² > 0.90)
- ✅ **15+ teknik gösterge:** Otomatik hesaplama
- ✅ **LazyPredict:** 40+ model otomatik test
- ✅ **Backtesting:** Tarihsel performans analizi

### 📈 Veri Analizi
- ✅ 5 yıllık tarihsel veri (Yahoo Finance)
- ✅ 26 hisse senedi (BIST-30 + S&P 500)
- ✅ 11+ EDA görselleştirme
- ✅ Korelasyon & volatilite analizi

---

## 📁 Proje Yapısı

```
exchangetrack/
│
├── 📄 app.py                          ⭐ STREAMLIT DASHBOARD
├── 📄 main.py                         📥 Veri toplama
├── 📄 run_eda.py                      📊 EDA analizi
├── 📄 setup_project.py                🔧 Proje kurulum
├── 📄 requirements.txt
├── 📄 README.md

│
├── 📁 data/
│   ├── raw/                           Ham CSV dosyaları (26 hisse)
│   ├── processed/
│   └── technical/                     Teknik göstergeli veriler
│
├── 📁 src/                            Kütüphane kodu
│   ├── data/
│   │   └── collector.py               StockDataCollector sınıfı
│   ├── analysis/
│   │   ├── eda.py                     ExploratoryDataAnalysis
│   │   └── technical.py               TechnicalAnalysis
│   ├── models/
│   │   ├── lazy_model_selector.py     LazyPredict wrapper
│   │   └── trainer.py                 Model eğitim
│   └── utils/
│       ├── visualization.py           Görselleştirme (Plotly, Matplotlib)
│       └── indicators.py              15+ teknik gösterge
│
├── 📁 models/                         Kaydedilmiş modeller (.pkl)
│   ├── AAPL_ridge_model.pkl
│   ├── GARAN_IS_lassolars_model.pkl
│   └── ... (16+ model dosyası)
│
├── 📁 outputs/
│   ├── eda_charts/                    11+ EDA grafiği
│   ├── lazy_predict/                  Model test sonuçları
│   └── reports/                       Analiz raporları
│
├── 📁 tests/
│   └── test_models.py                 Unit testler
│
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

---

## 🎯 Desteklenen Hisseler

### 🇹🇷 BIST-30 (11 hisse)
THYAO.IS, AKBNK.IS, GARAN.IS, ISCTR.IS, EREGL.IS, SAHOL.IS, KCHOL.IS, TUPRS.IS, PETKM.IS, SISE.IS, ASELS.IS

### 🇺🇸 S&P 500 (10 hisse)
AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, WMT

---

## 📊 Model Performansı

### Regression Modelleri (Production Ready)

| Hisse | Model | R² Score | RMSE | MAPE |
|-------|-------|----------|------|------|
| GARAN_IS | LassoLarsCV | **0.9410** | 0.234 | 2.18% |
| AAPL | Ridge | **0.9385** | 1.245 | 1.89% |
| MSFT | HuberRegressor | **0.9799** | 0.856 | 1.54% |
| THYAO_IS | LinearRegression | **0.8980** | 0.412 | 2.67% |

### Backtest Sonuçları (1 Yıl)

| Hisse | Getiri | Sharpe | Max DD | Win Rate |
|-------|--------|--------|--------|----------|
| GARAN_IS | **+37.68%** 🏆 | 1.12 | -25.29% | 66.7% |
| AAPL | +5.45% | 0.33 | -28.67% | 75.0% |

---

## 💻 Kullanım Komutları

### 1️⃣ Dashboard (Main)
```bash
streamlit run app.py
```
**Özellikleri:**
- 💰 Mevcut fiyat + Yarın tahmini
- 📈 BUY/SELL/HOLD sinyali
- 🔧 15+ teknik gösterge
- 📊 Backtest metrikleri (Sharpe, Max DD, Win Rate)

### 2️⃣ Veri Güncelle
```bash
python main.py
```
Tüm 26 hisse için 5 yıllık veri indir → `data/raw/*.csv`

### 3️⃣ EDA Analizi
```bash
python run_eda.py
```
11+ görselleştirme oluştur → `outputs/eda_charts/*.png`

### 4️⃣ Model Test (LazyPredict - 40+ model)
```bash
python run_lazy_predict.py
```
Otomatik model keşfi → `outputs/lazy_predict/*.csv`

### 5️⃣ Best Model Eğit
```bash
python train_best_models.py
```
Regression modellerini eğit → `models/*.pkl`

### 6️⃣ Sonuçları Analiz Et
```bash
python analyze_lazy_results.py
```
LazyPredict sonuçlarını analiz → `outputs/reports/`

### 7️⃣ Testler Çalıştır
```bash
pytest tests/ -v
```

---

## 🔧 Teknik Göstergeler (15+)

### Momentum
- **RSI (14)** - Overbought/Oversold
- **MACD** - Trend değişimi
- **Stochastic %K/%D** - Momentum
- **Williams %R** - Baskı göstergesi

### Trend
- **SMA (20, 50, 200)** - Hareketli ortalama
- **EMA (12, 26)** - Üstel ortalama
- **Pivot Points** - Destek/Direnç

### Volatilite
- **Bollinger Bands** - Fiyat aralığı
- **ATR (14)** - Gerçek aralık
- **BB Position** - Bant içi konum

### Hacim
- **OBV** - Birikimli hacim
- **MFI (14)** - Para akışı endeksi
- **Volume Ratio** - Hacim oranı

### Sinyal Üretimi
Çok göstergeli ağırlıklı scoring: **BUY (≥0.5)** | **SELL (≤-0.5)** | **HOLD**

---

## 🛠️ Teknoloji Stack

| Kategori | Teknoloji |
|----------|-----------|
| **Backend** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, LazyPredict |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Data Source** | Yahoo Finance API |
| **Deployment** | Docker, Docker Compose |

---

## 📚 Geliştirme Aşamaları (13 Hafta)

| Hafta | Aşama | Durum |
|-------|-------|-------|
| 1-2 | Veri toplama ve temizleme | ✅ |
| 3-4 | Keşifsel veri analizi (EDA) | ✅ |
| 5-7 | Teknik göstergeler (15+) | ✅ |
| 8-9 | Makine öğrenmesi modellemesi | ✅ |
| 10-12 | Streamlit web uygulaması | ✅ |
| 13 | Dokümantasyon & sunum | ✅ |

---

## 🎓 Önemli Bulgular

### Regression > Classification
- **Regression:** R² > 0.90 (Çok başarılı) ✅
- **Classification:** F1 Score < 0.70 (Düşük) ❌
- **Sonuç:** Fiyat tahmini, sinyal sınıflandırmasından çok daha iyi

### Piyasa Farkları
| Özellik | BIST-30 | S&P 500 |
|---------|---------|---------|
| Volatilite | 2.34% | 1.45% |
| Threshold | ±2% | ±1% |
| Karakteri | Yüksek volatil | Daha istikrarlı |

### En Başarılı Model: GARAN_IS
```
LassoLarsCV
R² = 0.9410 (Mükemmel!)
Backtest: +37.68% getiri, Sharpe = 1.12
```

---

## ⚠️ Yasal Uyarı

```
⚠️ DİSCLAİMER:
Bu sistem SADECE eğitim ve araştırma amaçlıdır.
❌ Finansal yatırım tavsiyesi DEĞILDIR
❌ Profesyonel danışmanlık yerine geçmez
✅ Algoritmalık ticaret eğitimi için tasarlandı

Gerçek para ile işlem yapmadan:
→ Profesyonel danışmanla konuşun
→ Kendi risk yönetimi yapın
→ Backtest sonuçlarını doğrulayın
```

---

## 🚀 Deployment

### Local
```bash
streamlit run app.py
```

### Docker
```bash
docker-compose up -d
# Tarayıcı: http://localhost:8501
```

---

## 📞 İletişim

- **GitHub Issues:** Bug report ve öneriler
- **LinkedIn:** https://www.linkedin.com/in/halil-o-a3a75b233/

---

## 📜 Lisans

MIT License - [LICENSE](LICENSE) dosyasına bakın

---

## 🙏 Teşekkürler

Açık kaynak kütüphanelere:
- yfinance (Yahoo Finance API)
- pandas, numpy (Veri işleme)
- scikit-learn (Makine öğrenmesi)
- streamlit (Web framework)
- plotly (Grafikler)

---

<div align="center">

**Made with ❤️ by Halil Öztekin**

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!

[GitHub](https://github.com/hoztekin) • [LinkedIn](https://www.linkedin.com/in/halil-o-a3a75b233/)

**Status:** ✅ Production Ready | Last Updated: November 2025

</div>