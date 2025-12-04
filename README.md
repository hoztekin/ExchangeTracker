# 📊 ExchangeTracker - Borsa Trend Analizi ve Tahmin Sistemi

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red)
![Status](https://img.shields.io/badge/Status-✅%20Production-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)

**Makine öğrenmesi ve teknik analiz kullanarak BIST-30 ve S&P 500 hisse senetlerini analiz eden ve tahmin eden profesyonel fintech sistemi.**

> 🎯 **13 haftalık akademik proje → Production-ready deployment + Otomatik pipeline**

<div align="center">

[🌐 Live Demo](https://exchangetrack.haliloztekin.com) • [📖 Dokümantasyon](#proje-yapısı) • [🤝 Katkıda Bulun](#katkıda-bulunma)

</div>

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum
```bash
# Repo'yu klonla
git clone https://github.com/hoztekin/ExchangeTracker
cd ExchangeTracker

# Virtual environment oluştur
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. Streamlit Dashboard'u Çalıştır
```bash
streamlit run app.py
```
**Tarayıcı otomatik açılır:** `http://localhost:8501`

### 3. Docker ile Production Deploy
```bash
# Docker Compose ile çalıştır
docker-compose up -d

# Logları takip et
docker-compose logs -f
```

**🌐 Production URL:** [https://exchangetrack.haliloztekin.com](https://exchangetrack.haliloztekin.com)

---

## ✨ Temel Özellikler

### 📊 Interactive Dashboard
- ✅ **Gerçek zamanlı tahminler:** Yarının kapanış fiyatı tahmini
- ✅ **BUY/SELL/HOLD sinyalleri:** Dinamik threshold'lar (US: ±1%, TR: ±2%)
- ✅ **15+ teknik gösterge:** RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic
- ✅ **Backtest metrikleri:** Sharpe Ratio, Maximum Drawdown, Win Rate
- ✅ **İnteraktif grafikler:** Plotly ile zoom, pan, hover detayları
- ✅ **Çoklu hisse analizi:** 10 BIST-30 + 10 S&P 500 hisse senedi
- ✅ **Pipeline kontrolü:** Manuel veri güncelleme ve model eğitimi

### 🤖 Makine Öğrenmesi
- ✅ **High-performance regression:** Ridge, LassoLarsCV (R² > 0.90)
- ✅ **LazyPredict entegrasyonu:** 40+ model otomatik test ve karşılaştırma
- ✅ **Akıllı feature engineering:** 15+ teknik gösterge + lag features
- ✅ **Backtesting simülasyonu:** Tarihsel performans doğrulama
- ✅ **Model persistence:** Eğitilmiş modeller .pkl formatında saklanır
- ✅ **Otomatik model seçimi:** En iyi performans gösteren model kullanılır

### 🔄 Otomasyon Pipeline
- ✅ **Günlük otomatik güncelleme:** Her gün saat 02:00'da veri güncelleme
- ✅ **Akıllı model yeniden eğitimi:** R² < 0.85 olduğunda otomatik retrain
- ✅ **State management:** pipeline_state.json ile durum takibi
- ✅ **Manuel tetikleme:** Dashboard'dan "Veri Güncelle" / "Model Eğit" butonları
- ✅ **Error handling & logging:** Hata durumlarında detaylı loglama
- ✅ **Graceful degradation:** Pipeline çökse bile sistem çalışmaya devam eder

### 📈 Veri Analizi
- ✅ **5 yıllık tarihsel veri:** Yahoo Finance API
- ✅ **20 hisse senedi:** 10 BIST-30 + 10 S&P 500
- ✅ **Kapsamlı EDA:** Korelasyon, volatilite, trend analizi
- ✅ **Görselleştirmeler:** Candlestick, volume, teknik göstergeler

---

## 📁 Proje Yapısı

```
ExchangeTracker/
│
├── 📄 app.py                          ⭐ STREAMLIT DASHBOARD (Ana Uygulama)
│                                       • Dashboard UI ve state management
│                                       • Model yükleme ve tahmin
│                                       • Pipeline kontrolü (manuel tetikleme)
│                                       • Teknik gösterge grafikleri
│                                       • Backtest sonuçları görselleştirme
│
├── 📄 main.py                         📥 Veri Toplama (Data Collection)
│                                       • Yahoo Finance API ile veri çekme
│                                       • 20 hisse için 5 yıllık data
│                                       • Ham CSV kaydetme (data/raw/)
│
├── 📄 run_eda.py                      📊 EDA Çalıştırıcı (Exploratory Data Analysis)
│                                       • scripts/eda/descriptive_stats.py
│                                       • scripts/eda/price_analysis.py
│                                       • scripts/eda/volume_analysis.py
│                                       • scripts/eda/correlation_analysis.py
│                                       • scripts/eda/trend_analysis.py
│
├── 📄 run_technical_analysis.py       📈 Teknik Analiz Çalıştırıcı
│                                       • scripts/technical_analysis/indicators.py
│                                       • SMA, EMA, RSI, MACD, Bollinger, ATR hesaplama
│                                       • Technical CSV kaydetme (data/technical/)
│
├── 📄 run_lazy_predict.py             🤖 LazyPredict Model Test
│                                       • 40+ regression model otomatik test
│                                       • Model performans karşılaştırması
│                                       • outputs/lazy_predict_results.csv
│
├── 📄 requirements.txt                📦 Python Bağımlılıkları
├── 📄 README.md                       📖 Bu Dokümantasyon
├── 📄 Dockerfile                      🐳 Container Image Tanımı
├── 📄 docker-compose.yml              🐳 Multi-container Orchestration
├── 📄 LICENSE                         📜 MIT Lisansı
├── 📄 .gitignore                      🚫 Git Ignore Rules
├── 📄 pipeline_state.json             💾 Pipeline State (otomatik oluşur)
│
├── 📁 .venv/                          🐍 Virtual Environment (git ignore)
│
├── 📁 data/                           💾 VERİ DEPOLAMA
│   ├── raw/                           • Ham CSV dosyaları
│   │   ├── GARAN.IS.csv
│   │   ├── AAPL.csv
│   │   └── ... (20 dosya)
│   │
│   └── technical/                     • Teknik göstergeler eklenmiş
│       ├── GARAN.IS_technical.csv
│       ├── AAPL_technical.csv
│       └── ... (20 dosya)
│
├── 📁 models/                         🤖 EĞİTİLMİŞ ML MODELLERİ
│   │                                   Format: {TICKER}_{MODEL}_model.pkl
│   ├── GARAN_IS_lassolars_model.pkl   • LassoLarsCV (R² = 0.9410)
│   ├── AAPL_ridge_model.pkl           • Ridge (R² = 0.9385)
│   ├── MSFT_huber_model.pkl           • HuberRegressor (R² = 0.9799)
│   └── ... (20+ model dosyası)
│
├── 📁 outputs/                        📊 ANALİZ ÇIKTILARI
│   ├── backtest_report.txt            • Backtest performans raporu
│   └── lazy_predict_results.csv       • Model karşılaştırma tablosu
│
├── 📁 logs/                           📝 PİPELİNE LOGLARI (otomatik oluşur)
│   └── pipeline.log                   • Otomatik güncelleme kayıtları
│
├── 📁 pipeline/                       🔄 OTOMASYON SİSTEMİ
│   ├── __init__.py                    • Package init
│   │
│   ├── config.py                      ⚙️ Pipeline Konfigürasyonu
│   │                                   • BIST30_STOCKS = [...]
│   │                                   • SP500_STOCKS = [...]
│   │                                   • MIN_R2_SCORE = 0.85
│   │                                   • UPDATE_TIME = time(2, 0)
│   │                                   • RETRAIN_THRESHOLD_DAYS = 7
│   │
│   ├── scheduler.py                   ⏰ APScheduler Yönetimi
│   │                                   • PipelineScheduler sınıfı
│   │                                   • start() / stop() fonksiyonları
│   │                                   • manual_update_stock(ticker)
│   │                                   • manual_train_model(ticker)
│   │                                   • State management (JSON)
│   │
│   ├── data_updater.py                📥 Otomatik Veri Güncelleme
│   │                                   • DataUpdater sınıfı
│   │                                   • Yahoo Finance entegrasyonu
│   │                                   • Teknik gösterge hesaplama
│   │                                   • update_stock(ticker) fonksiyonu
│   │
│   └── model_trainer.py               🤖 Otomatik Model Eğitimi
│       │                               • ModelTrainer sınıfı
│       │                               • train_model(ticker, force_retrain)
│       │                               • LazyPredict entegrasyonu
│       │                               • Threshold-based retraining
│       │                               • Model performance monitoring
│
├── 📁 scripts/                        🔧 YARDIMCI SCRIPTLER
│   │
│   ├── 📁 eda/                        📊 EDA Modülleri
│   │   ├── __init__.py
│   │   ├── descriptive_stats.py      • Temel istatistikler
│   │   ├── price_analysis.py         • Fiyat analizi ve grafikler
│   │   ├── volume_analysis.py        • İşlem hacmi analizi
│   │   ├── correlation_analysis.py   • Korelasyon matrisleri
│   │   └── trend_analysis.py         • Trend ve volatilite
│   │
│   ├── 📁 technical_analysis/         📈 Teknik Analiz Modülleri
│   │   ├── __init__.py
│   │   └── indicators.py             • Tüm teknik göstergeler
│   │                                   calculate_sma(), calculate_ema()
│   │                                   calculate_rsi(), calculate_macd()
│   │                                   calculate_bollinger_bands()
│   │                                   calculate_atr(), calculate_obv()
│   │
│   ├── train_best_models.py           🎯 En İyi Modelleri Eğit
│   │                                   • BestModelTrainer sınıfı
│   │                                   • Ridge, LassoLarsCV, HuberRegressor
│   │                                   • Feature engineering
│   │                                   • Model kaydetme (pickle)
│   │
│   ├── backtest.py                    📊 Backtesting Simülasyonu
│   │                                   • Backtester sınıfı
│   │                                   • Trading stratejisi testi
│   │                                   • Sharpe Ratio, Max Drawdown
│   │                                   • Win rate hesaplama
│   │
│   └── analyze_lazy_results.py        📋 LazyPredict Analizi
│                                       • Model performans karşılaştırması
│                                       • outputs/lazy_predict_results.csv
│
└── 📁 tests/                          🧪 TEST MODÜLLERI
    └── test_models.py                 • Model test ve tahmin
                                       • ModelTester sınıfı
                                       • Yarın tahmini ve sinyal üretimi
```

### 🎯 Dosya Fonksiyonları

#### **Kök Dizin Python Dosyaları**
| Dosya | Açıklama | Çalıştırma |
|-------|----------|------------|
| **app.py** | Streamlit dashboard (Ana uygulama) | `streamlit run app.py` |
| **main.py** | Veri toplama (Yahoo Finance → raw CSV) | `python main.py` |
| **run_eda.py** | EDA analizlerini çalıştır | `python run_eda.py` |
| **run_technical_analysis.py** | Teknik göstergeleri hesapla | `python run_technical_analysis.py` |
| **run_lazy_predict.py** | 40+ model test et (LazyPredict) | `python run_lazy_predict.py` |

#### **Pipeline Modülleri** (`pipeline/`)
| Dosya | Açıklama |
|-------|----------|
| **config.py** | Hisse listeleri, eşik değerleri, zaman ayarları |
| **scheduler.py** | APScheduler ile otomatik zamanlama ve state management |
| **data_updater.py** | Yahoo Finance'ten veri çekme ve teknik gösterge ekleme |
| **model_trainer.py** | Model eğitimi, performans kontrolü ve retraining |

#### **Scripts Modülleri** (`scripts/`)
| Dosya | Açıklama | Dizin |
|-------|----------|-------|
| **train_best_models.py** | En iyi modelleri manuel eğit | `scripts/` |
| **backtest.py** | Trading stratejisi backtesting | `scripts/` |
| **analyze_lazy_results.py** | LazyPredict sonuçlarını analiz et | `scripts/` |
| **descriptive_stats.py** | Temel istatistikler (mean, std, min, max) | `scripts/eda/` |
| **price_analysis.py** | Fiyat grafikleri ve dağılımlar | `scripts/eda/` |
| **volume_analysis.py** | İşlem hacmi analizi | `scripts/eda/` |
| **correlation_analysis.py** | Korelasyon matrisi ve heatmap | `scripts/eda/` |
| **trend_analysis.py** | Trend ve volatilite analizi | `scripts/eda/` |
| **indicators.py** | Tüm teknik gösterge hesaplamaları | `scripts/technical_analysis/` |

#### **Test Modülleri** (`tests/`)
| Dosya | Açıklama |
|-------|----------|
| **test_models.py** | Model test, tahmin ve BUY/SELL/HOLD sinyal üretimi |

---

## 💻 Kullanım Komutları

### 🚀 Tam Workflow (Baştan Sona)

```bash
# 1️⃣ Veri toplama (5 yıllık tarihsel data)
python main.py

# 2️⃣ EDA analizi (görselleştirmeler)
python run_eda.py

# 3️⃣ Teknik göstergeleri hesapla
python run_technical_analysis.py

# 4️⃣ Model keşfi (40+ model test)
python run_lazy_predict.py

# 5️⃣ En iyi modelleri eğit
python scripts/train_best_models.py

# 6️⃣ Backtest simülasyonu
python scripts/backtest.py

# 7️⃣ Dashboard'u başlat
streamlit run app.py
```

### 📊 Sadece Dashboard (Production Kullanımı)

```bash
# Eğer models/ klasörü hazırsa direkt dashboard başlat
streamlit run app.py
```

**Dashboard'da şunları yapabilirsiniz:**
- 💰 Güncel fiyat ve yarın tahmini görüntüleme
- 📈 BUY/SELL/HOLD sinyal alma
- 🔧 15+ teknik gösterge grafiği inceleme
- 📊 Backtest performans metrikleri görme
- 🔄 Manuel veri güncelleme (pipeline varsa)
- 🤖 Manuel model eğitimi (pipeline varsa)

---

## 🎯 Desteklenen Hisseler

### 🇹🇷 BIST-30 (10 hisse)
```python
GARAN.IS    # Garanti Bankası
THYAO.IS    # Türk Hava Yolları
AKBNK.IS    # Akbank
EREGL.IS    # Ereğli Demir Çelik
TUPRS.IS    # Tüpraş
KCHOL.IS    # Koç Holding
SAHOL.IS    # Sabancı Holding
ASELS.IS    # Aselsan
SISE.IS     # Şişe Cam
TCELL.IS    # Turkcell
```

### 🇺🇸 S&P 500 (10 hisse)
```python
AAPL    # Apple Inc.
MSFT    # Microsoft Corp.
GOOGL   # Alphabet Inc.
AMZN    # Amazon.com Inc.
TSLA    # Tesla Inc.
META    # Meta Platforms Inc.
NVDA    # NVIDIA Corp.
JPM     # JPMorgan Chase & Co.
V       # Visa Inc.
WMT     # Walmart Inc.
```

---

## 📊 Model Performansı

### 🏆 Production Model Sonuçları

| Hisse | Model | R² Score (Test) | MAPE | Train Date | Status |
|-------|-------|-----------------|------|------------|--------|
| **GARAN.IS** | LassoLarsCV | **0.9410** | 2.18% | 2025-11-27 | ✅ Production |
| **AAPL** | Ridge | **0.9385** | 1.89% | 2025-11-27 | ✅ Production |
| **MSFT** | HuberRegressor | **0.9799** | 1.54% | 2025-11-27 | ✅ Production |
| **THYAO.IS** | LinearRegression | **0.8980** | 2.67% | 2025-11-27 | ✅ Production |

> **Not:** R² > 0.90 skoru, modelin varyansın %90'ından fazlasını açıklayabildiğini gösterir.

### 💰 Backtest Sonuçları (1 Yıl Simülasyonu)

| Hisse | Toplam Getiri | Sharpe Ratio | Max Drawdown | İşlem Sayısı | Kazanma Oranı |
|-------|---------------|--------------|--------------|--------------|---------------|
| **GARAN.IS** 🏆 | **+37.68%** | 1.12 | -25.29% | 18 | 66.7% |
| **AAPL** | +5.45% | 0.33 | -28.67% | 8 | 75.0% |

**Backtest Parametreleri:**
- 💵 Başlangıç sermayesi: $10,000
- 📊 İşlem başına yatırım: Sermayenin %95'i
- 💳 Komisyon: İşlem başına %0.1
- 🎯 Sinyal threshold'ları: US hisseleri ±1%, Türk hisseleri ±2%
- 📅 Test periyodu: Son 1 yıl (252 işlem günü)

---

## 🐳 Docker Deployment

### 📄 docker-compose.yml
```yaml
version: '3.8'

services:
  exchangetracker:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./pipeline_state.json:/app/pipeline_state.json
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 📄 Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Gerekli klasörleri oluştur
RUN mkdir -p /app/logs /app/data/raw /app/data/technical /app/models /app/outputs

# Port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Streamlit'i başlat
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### 🚢 Deployment Adımları

```bash
# 1. Docker image build et
docker-compose build

# 2. Container'ı arka planda başlat
docker-compose up -d

# 3. Logları canlı takip et
docker-compose logs -f

# 4. Container durumunu kontrol et
docker-compose ps

# 5. Container'a shell ile bağlan (debug için)
docker-compose exec exchangetracker bash

# 6. Container'ı durdur
docker-compose down

# 7. Volume'ları da sil (tüm veriyi sil)
docker-compose down -v
```

### 🌐 Domain Yapılandırması

**Cloudflare DNS Ayarları:**
```
Type: A
Name: exchangetrack
Content: 128.140.73.107
Proxy: ✅ Proxied (Orange Cloud)
TTL: Auto
```

**Nginx Reverse Proxy (Sunucuda):**
```nginx
server {
    listen 80;
    server_name exchangetrack.haliloztekin.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 📊 Portainer Management

**Portainer Stacks:**
```yaml
Name: exchangetracker
Stack file: docker-compose.yml
Env variables:
  - STREAMLIT_SERVER_PORT=8501
  - STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

**Container Bilgileri:**
- **Image:** exchangetracker:latest
- **Port:** 8501:8501
- **Restart Policy:** unless-stopped
- **Volumes:** data/, models/, logs/, pipeline_state.json
- **Health Check:** ✅ Enabled (30s interval)

---

## 🔧 Pipeline Yapılandırması

### ⚙️ config.py - Temel Ayarlar

```python
# Scheduler ayarları
SCHEDULER_ENABLED = True
UPDATE_TIME = time(2, 0)  # Her gün 02:00'da çalış
TIMEZONE = 'Europe/Istanbul'

# Model eğitim parametreleri
MIN_R2_SCORE = 0.85  # Bu değerin altına düşerse yeniden eğit
RETRAIN_THRESHOLD_DAYS = 7  # X gün geçtiyse performans kontrolü yap

# Hisse listeleri
BIST30_STOCKS = ['GARAN.IS', 'THYAO.IS', 'AKBNK.IS', ...]
SP500_STOCKS = ['AAPL', 'MSFT', 'GOOGL', ...]

# Teknik göstergeler
INDICATORS = [
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
    'ATR', 'OBV', 'Stochastic'
]
```

### 🔄 Pipeline Çalışma Mantığı

```
┌─────────────────────────────────────────────────────────────┐
│         GÜNLÜK OTOMATİK ÇALIŞTIRMA (02:00 İstanbul)        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Data Updater   │
              │  ─────────────  │
              │  • Yahoo Finance│
              │  • Raw CSV      │
              │  • Technical    │
              │    Indicators   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Model Trainer   │
              │ ─────────────── │
              │ • R² kontrolü   │
              │ • Retrain logic │
              │ • LazyPredict   │
              │ • Save .pkl     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ State Update    │
              │ ─────────────── │
              │ • JSON kaydet   │
              │ • Timestamp     │
              │ • Performance   │
              └─────────────────┘
```

**Akıllı Yeniden Eğitim Mantığı:**
1. ✅ Mevcut model var mı? → Performansını kontrol et
2. ❌ R² < 0.85 → Yeniden eğit
3. 📅 Son eğitimden 7+ gün geçti mi? → Kontrol et
4. 🆚 Yeni model daha iyi mi? → Değiştir

---

## 📊 Teknik Göstergeler

### 🔧 Hesaplanan Göstergeler (indicators.py)

| Gösterge | Açıklama | Kullanım |
|----------|----------|----------|
| **SMA (Simple Moving Average)** | Basit hareketli ortalama | Trend takibi |
| **EMA (Exponential Moving Average)** | Üssel ağırlıklı ortalama | Kısa vadeli trend |
| **RSI (Relative Strength Index)** | Momentum osilatörü (0-100) | Aşırı alım/satım |
| **MACD** | Momentum göstergesi | Trend dönüşü |
| **Bollinger Bands** | Volatilite bantları | Fiyat aralığı |
| **ATR (Average True Range)** | Ortalama volatilite | Risk ölçümü |
| **OBV (On-Balance Volume)** | Hacim bazlı momentum | Akım analizi |
| **Stochastic Oscillator** | Momentum osilatörü | Aşırı alım/satım |

### 📈 Feature Engineering

**Model için kullanılan özellikler:**
```python
features = [
    'open', 'high', 'low', 'close', 'volume',
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_Signal', 
    'BB_Upper', 'BB_Lower', 'BB_Middle',
    'ATR', 'OBV', 'Stochastic',
    'price_change_1d', 'price_change_5d',
    'momentum_10', 'volatility_20', 'volume_ratio'
]

# Target: Yarının kapanış fiyatı
target = df['close'].shift(-1)
```

---

## 🚦 BUY/SELL/HOLD Sinyalleri

### 📊 Sinyal Üretim Algoritması

```python
def generate_signal(current_price, predicted_price, ticker):
    change_pct = (predicted_price - current_price) / current_price * 100
    
    # Dinamik threshold (Türk hisseleri daha volatil)
    threshold = 2.0 if '.IS' in ticker else 1.0
    
    if change_pct >= threshold:
        return 'BUY 📈'
    elif change_pct <= -threshold:
        return 'SELL 📉'
    else:
        return 'HOLD ⏸️'
```

### 🎯 Threshold Değerleri

| Piyasa | Threshold | Açıklama |
|--------|-----------|----------|
| **🇺🇸 US Stocks** | ±1.0% | Düşük volatilite |
| **🇹🇷 BIST Stocks** | ±2.0% | Yüksek volatilite |

**Örnek:**
- AAPL: +0.8% → HOLD (1% threshold)
- GARAN.IS: +2.3% → BUY (2% threshold)
- TSLA: -1.5% → SELL (1% threshold)

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. **Fork** edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. **Commit** edin (`git commit -m 'feat: Add amazing feature'`)
4. **Push** edin (`git push origin feature/amazing-feature`)
5. **Pull Request** açın

### 📋 Coding Standards

- ✅ PEP 8 Python style guide
- ✅ Type hints kullanımı
- ✅ Docstring'ler (fonksiyon açıklamaları)
- ✅ Logging ile hata takibi
- ✅ Try-except ile error handling

### 🛣️ Geliştirme Yol Haritası

- [ ] **Real-time veri akışı:** WebSocket ile canlı fiyatlar
- [ ] **Daha fazla teknik gösterge:** Ichimoku, Fibonacci Retracement
- [ ] **Sentiment analizi:** Twitter, Reddit API entegrasyonu
- [ ] **Portfolio optimizasyonu:** Markowitz Mean-Variance
- [ ] **E-posta/SMS bildirimleri:** Sinyal alarmları
- [ ] **Multi-timeframe analizi:** 1h, 4h, 1d, 1w
- [ ] **Deep Learning modeller:** LSTM, GRU, Transformer
- [ ] **Mobil uygulama:** React Native veya Flutter

---

## ⚠️ Yasal Uyarı

**⚠️ DİKKAT:** Bu yazılım yalnızca **eğitim ve araştırma** amaçlıdır.

- ❌ **Finansal tavsiye değildir**
- ❌ **Yatırım garantisi vermez**
- ❌ **Gerçek parayla işlem yapmadan önce profesyonel danışman görüşü alın**
- ⚠️ **Geçmiş performans gelecek getiriyi garanti etmez**
- ⚠️ **Borsa yatırımları risk içerir, sermaye kaybı yaşayabilirsiniz**
- ⚠️ **Kullanıcılar kendi kararlarından sorumludur**

**📜 Geliştirici, bu yazılımın kullanımından kaynaklanan herhangi bir finansal kayıptan sorumlu tutulamaz.**

Yatırım kararlarınızı alırken:
- 📊 Kendi araştırmanızı yapın (DYOR - Do Your Own Research)
- 💼 Profesyonel finansal danışman görüşü alın
- 🎯 Risk toleransınızı belirleyin
- 💰 Sadece kaybetmeyi göze alabileceğiniz miktarla yatırım yapın

---

## 📜 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır.

```
MIT License

Copyright (c) 2025 Halil Öztekin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Tam lisans metni için LICENSE dosyasına bakın]
```

Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 👨‍💻 Geliştirici

**Halil Öztekin**
- 🎓 **Üniversite:** Konya Teknik Üniversitesi - Bilgisayar Mühendisliği
- 📧 **Email:** hoztekin81@gmail.com
- 🔗 **GitHub:** [@hoztekin](https://github.com/hoztekin)
- 💼 **LinkedIn:** [Halil Öztekin](https://www.linkedin.com/in/halil-o-a3a75b233/)
- 🌐 **Website:** [haliloztekin.com](https://haliloztekin.com)

---

## 🙏 Teşekkürler

Bu projeyi hayata geçiren açık kaynak projelere ve teknolojilere teşekkürler:

- **[Streamlit](https://streamlit.io/)** - Dashboard framework
- **[Yahoo Finance](https://finance.yahoo.com/)** - Finansal veri API
- **[LazyPredict](https://github.com/shankarpandala/lazypredict)** - Otomatik model seçimi
- **[Plotly](https://plotly.com/)** - İnteraktif grafikler
- **[Scikit-learn](https://scikit-learn.org/)** - Makine öğrenmesi
- **[Pandas](https://pandas.pydata.org/)** - Veri manipülasyonu
- **[NumPy](https://numpy.org/)** - Sayısal hesaplama
- **[APScheduler](https://apscheduler.readthedocs.io/)** - Görev zamanlama
- **[Docker](https://www.docker.com/)** - Containerization
- **[Cloudflare](https://www.cloudflare.com/)** - DNS & Proxy

---

## 📊 Proje İstatistikleri

- 📅 **Başlangıç:** Ekim 2024
- 📅 **Production Deploy:** Kasım 2024
- ⏱️ **Geliştirme Süresi:** 13 hafta
- 📝 **Toplam Kod Satırı:** ~5000+ satır Python
- 📊 **Veri Sayısı:** 20 hisse × 5 yıl × 252 gün = ~25,000 veri noktası
- 🤖 **Model Sayısı:** 20+ eğitilmiş model
- 📈 **Teknik Gösterge:** 15+ gösterge
- 🧪 **Test Edilen Model:** 40+ (LazyPredict)

---

## 🎓 Akademik Bilgiler

**Proje Türü:** Bitirme Projesi (Capstone Project)  
**Üniversite:** Konya Teknik Üniversitesi  
**Bölüm:** Bilgisayar Mühendisliği  
**Dönem:** 2024-2025 Güz Dönemi  
**Süre:** 13 hafta  
**Danışman:** Doç.Drç Sait Ali UYMAZ (sauymaz@ktun.edu.tr)

**Kullanılan Teknolojiler:**
- Python 3.9+
- Streamlit 1.38+
- Scikit-learn
- Pandas, NumPy
- Yahoo Finance API
- Docker & Docker Compose
- Plotly
- APScheduler

**Hedefler:**
- ✅ Real-world veri ile çalışma deneyimi
- ✅ Makine öğrenmesi model geliştirme
- ✅ Production deployment (Docker)
- ✅ API entegrasyonu (Yahoo Finance)
- ✅ Web uygulama geliştirme (Streamlit)
- ✅ Otomasyon pipeline (Scheduler)

---

## 🔗 Faydalı Linkler

- 🌐 **Live Demo:** [https://exchangetrack.haliloztekin.com](https://exchangetrack.haliloztekin.com)
- 📦 **GitHub Repo:** [https://github.com/hoztekin/ExchangeTracker](https://github.com/hoztekin/ExchangeTracker)
- 📖 **Streamlit Docs:** [https://docs.streamlit.io](https://docs.streamlit.io)
- 📊 **Yahoo Finance API:** [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)
- 🤖 **LazyPredict:** [https://github.com/shankarpandala/lazypredict](https://github.com/shankarpandala/lazypredict)
- 🐳 **Docker Docs:** [https://docs.docker.com](https://docs.docker.com)

---

<div align="center">

**⭐ Bu projeyi beğendiyseniz yıldız (star) vermeyi unutmayın!**

[![GitHub stars](https://img.shields.io/github/stars/hoztekin/ExchangeTracker?style=social)](https://github.com/hoztekin/ExchangeTracker)
[![GitHub forks](https://img.shields.io/github/forks/hoztekin/ExchangeTracker?style=social)](https://github.com/hoztekin/ExchangeTracker/fork)
[![GitHub watchers](https://img.shields.io/github/watchers/hoztekin/ExchangeTracker?style=social)](https://github.com/hoztekin/ExchangeTracker)

---

[⬆ Başa Dön](#-exchangetracker---borsa-trend-analizi-ve-tahmin-sistemi)

</div>