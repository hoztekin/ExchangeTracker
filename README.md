# 📊 ExchangeTracker - Borsa Trend Analizi ve Tahmin Sistemi

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![Status](https://img.shields.io/badge/Status-✅%20Production-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

**Makine öğrenmesi ve teknik analiz kullanarak BIST-30 ve S&P 500 hisse senetlerini analiz eden ve tahmin eden profesyonel fintech sistemi.**

> 🎯 **13 haftalık akademik proje → Production-ready sistem + Otomatik pipeline**

<div align="center">

[🌐 Demo](http://128.140.73.107:8501) • [📖 Dokümantasyon](#proje-yapısı) • [🤝 Katkıda Bulun](#katkıda-bulunma)

</div>

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum
```bash
# Repo klonla
git clone https://github.com/hoztekin/ExchangeTracker
cd ExchangeTracker

# Sanal ortam oluştur
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. Streamlit Dashboard Çalıştır
```bash
streamlit run app.py
```
**Tarayıcı otomatik açılır:** `http://localhost:8501`

### 3. Docker ile Deploy (Production)
```bash
# Docker compose ile çalıştır
docker-compose up -d

# Logları takip et
docker-compose logs -f
```
**Production URL:** `http://128.140.73.107:8501`

---

## ✨ Temel Özellikler

### 📊 Interactive Dashboard
- ✅ **Gerçek zamanlı tahminler:** Yarının kapanış fiyatı tahmini
- ✅ **BUY/SELL/HOLD sinyalleri:** Dinamik threshold'lar (US: ±1%, TR: ±2%)
- ✅ **15+ teknik gösterge:** RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic
- ✅ **Backtest metrikleri:** Sharpe Ratio, Maximum Drawdown, Win Rate
- ✅ **İnteraktif grafikler:** Plotly ile zoom, pan, hover detayları
- ✅ **Çoklu hisse analizi:** 10 BIST-30 + 10 S&P 500 hisse senedi

### 🤖 Makine Öğrenmesi
- ✅ **Best-in-class regression modelleri:** Ridge, LassoLarsCV (R² > 0.90)
- ✅ **LazyPredict entegrasyonu:** 40+ model otomatik test ve karşılaştırma
- ✅ **Akıllı feature engineering:** 15+ teknik gösterge otomatik hesaplama
- ✅ **Backtesting simülasyonu:** Tarihsel performans doğrulama
- ✅ **Model persistence:** Eğitilmiş modeller .pkl formatında saklanır

### 🔄 Otomasyon Pipeline (Opsiyonel)
- ✅ **Günlük otomatik güncelleme:** Her gün saat 02:00'da veri güncelleme
- ✅ **Akıllı model yeniden eğitimi:** R² < 0.85 olduğunda otomatik retrain
- ✅ **State management:** pipeline_state.json ile durum takibi
- ✅ **Manuel tetikleme:** Dashboard'dan "Veri Güncelle" / "Model Eğit" butonları
- ✅ **Error handling & logging:** Hata durumlarında detaylı loglama
- ✅ **Graceful degradation:** Pipeline olmadan da sistem çalışır

### 📈 Veri Analizi
- ✅ 5 yıllık tarihsel veri (Yahoo Finance API)
- ✅ 20 hisse senedi (10 BIST-30 + 10 S&P 500)
- ✅ Kapsamlı EDA görselleştirmeleri
- ✅ Korelasyon, volatilite ve trend analizi

---

## 📁 Proje Yapısı

```
ExchangeTracker/
│
├── 📄 app.py                          ⭐ STREAMLIT DASHBOARD (Ana Uygulama)
├── 📄 main.py                         📥 Ana veri toplama scripti
├── 📄 run_eda.py                      📊 EDA analizi çalıştırıcı
├── 📄 run_lazy_predict.py             🤖 LazyPredict model test
├── 📄 run_technical_analysis.py       📈 Teknik analiz çalıştırıcı
├── 📄 requirements.txt                📦 Python bağımlılıkları
├── 📄 README.md                       📖 Dokümantasyon
├── 📄 Dockerfile                      🐳 Container image tanımı
├── 📄 docker-compose.yml              🐳 Multi-container orchestration
├── 📄 License                         📜 MIT Lisansı
├── 📄 .gitignore
│
├── 📁 .venv/                          🐍 Virtual Environment (library root)
│
├── 📁 data/                           💾 VERİ DEPOLAMA
│   ├── raw/                           Ham CSV dosyaları (orijinal Yahoo Finance)
│   │   ├── GARAN.IS.csv               
│   │   ├── AAPL.csv
│   │   └── ... (20 dosya)
│   │
│   └── technical/                     Teknik göstergeler eklenmiş veriler
│       ├── GARAN.IS_technical.csv     SMA, EMA, RSI, MACD, Bollinger, ATR, vb.
│       ├── AAPL_technical.csv
│       └── ... (20 dosya)
│
├── 📁 models/                         🤖 EĞİTİLMİŞ ML MODELLERİ
│   ├── GARAN_IS_lassolars_model.pkl   Model + scaler + metadata
│   ├── AAPL_ridge_model.pkl
│   ├── MSFT_ridge_model.pkl
│   └── ... (20+ model dosyası)
│
├── 📁 outputs/                        📊 ANALİZ ÇIKTILARI
│   ├── backtest_report.txt            Backtest performans raporu
│   └── lazy_predict_results.csv       Model karşılaştırma tablosu
│
├── 📁 logs/                           📝 PİPELİNE LOGLARI (oluşturulur)
│   └── pipeline.log                   Otomatik güncelleme kayıtları
│
├── 📁 pipeline/                       🔄 OTOMASYON SİSTEMİ (Opsiyonel)
│   ├── __init__.py
│   ├── config.py                      Pipeline yapılandırması
│   │                                  - Hisse listesi (BIST30_STOCKS, SP500_STOCKS)
│   │                                  - Eğitim parametreleri (MIN_R2_SCORE, RETRAIN_THRESHOLD_DAYS)
│   │                                  - Scheduler ayarları (UPDATE_TIME, TIMEZONE)
│   │
│   ├── scheduler.py                   APScheduler ile zamanlama
│   │                                  - Günlük otomatik çalıştırma
│   │                                  - Manuel tetikleme fonksiyonları
│   │                                  - State management
│   │
│   ├── data_updater.py                Otomatik veri güncelleme
│   │                                  - Yahoo Finance API entegrasyonu
│   │                                  - Teknik gösterge hesaplama
│   │                                  - Hata yönetimi
│   │
│   └── model_trainer.py               Otomatik model eğitimi
│                                      - LazyPredict ile model seçimi
│                                      - Model performans değerlendirme
│                                      - Threshold-based retraining
│
├── 📄 pipeline_state.json             📊 PİPELİNE DURUM DOSYASI
│                                      {
│                                        "last_update": "2025-11-27 02:00:00",
│                                        "next_scheduled": "2025-11-28 02:00:00",
│                                        "status": "idle",
│                                        "stocks": {
│                                          "GARAN.IS": {
│                                            "last_data_update": "...",
│                                            "last_model_update": "...",
│                                            "r2_score": 0.9410,
│                                            "model_status": "good"
│                                          }
│                                        }
│                                      }
│
├── 📁 scripts/                        🛠️ YARDIMCI SCRİPTLER
│   ├── outputs/                       Script çıktıları
│   │   ├── __init__.py
│   │   ├── analyze_lazy_results.py    LazyPredict sonuç analizi
│   │   ├── backtest.py                Backtest simülasyonu
│   │   └── train_best_models.py       En iyi modelleri eğit
│   │
│   └── src/                           Kaynak kod modülleri
│       ├── analysis/                  Analiz modülleri
│       │   └── __init__.py
│       │
│       ├── data/                      Veri işleme modülleri
│       │   └── __init__.py
│       │
│       ├── models/                    Model eğitim modülleri
│       │   └── __init__.py
│       │
│       └── utils/                     Yardımcı fonksiyonlar
│           └── __init__.py
│
├── 📁 streamlit_app/                  📱 STREAMLIT UYGULAMASI (alternatif yapı)
│
└── 📁 tests/                          🧪 TEST DOSYALARI
    └── (test dosyaları)
```

### 📌 Klasör Detayları

#### **`data/`** - Veri Depolama
- **`raw/`**: Yahoo Finance'ten çekilen ham CSV dosyaları
  - Kolonlar: Date, Open, High, Low, Close, Adj Close, Volume
  - Format: `{TICKER}.csv` (örn: GARAN_IS.csv, AAPL.csv)
  
- **`technical/`**: Teknik göstergeler hesaplanmış veriler
  - Ek kolonlar: SMA_20, SMA_50, EMA_12, EMA_26, RSI, MACD, MACD_Signal, BB_Upper, BB_Lower, ATR, OBV, Stochastic
  - Format: `{TICKER}_technical.csv`

#### **`models/`** - Makine Öğrenmesi Modelleri
- Pickle formatında kaydedilmiş model dosyaları
- Format: `{TICKER}_{MODEL_NAME}_model.pkl`
- İçerik yapısı:
  ```python
  {
      'model': trained_model,              # Scikit-learn model objesi
      'scaler': StandardScaler(),          # Feature normalization
      'feature_columns': [...],            # Eğitimde kullanılan özellikler
      'model_name': 'Ridge',               # Model ismi
      'r2_score': 0.9385,                  # Test R² skoru
      'mape': 1.89,                        # Mean Absolute Percentage Error
      'trained_date': '2025-11-27'         # Eğitim tarihi
  }
  ```

#### **`pipeline/`** - Otomasyon Sistemi (Opsiyonel)
> **Not:** Bu klasör opsiyoneldir. Pipeline olmadan da sistem tam çalışır.

- **`config.py`**: Tüm pipeline yapılandırması
  - Hisse listeleri (BIST30_STOCKS, SP500_STOCKS)
  - Model eğitim parametreleri (MIN_R2_SCORE, RETRAIN_THRESHOLD_DAYS)
  - Scheduler ayarları (UPDATE_TIME, TIMEZONE)
  - Teknik gösterge listesi (INDICATORS)

- **`scheduler.py`**: APScheduler ile otomatik zamanlama
  - `start()`: Scheduler'ı başlat
  - `manual_update_stock(ticker)`: Tek hisse için manuel güncelleme
  - `manual_train_model(ticker)`: Tek hisse için manuel eğitim
  - State management (pipeline_state.json)

- **`data_updater.py`**: Otomatik veri güncelleme
  - Yahoo Finance API entegrasyonu
  - Teknik gösterge hesaplama
  - Hata yönetimi ve retry logic

- **`model_trainer.py`**: Otomatik model eğitimi
  - LazyPredict ile en iyi modeli bulma
  - Mevcut model performans kontrolü
  - Threshold-based retraining (R² < 0.85)

#### **`pipeline_state.json`** - Durum Takibi
Sistemin mevcut durumunu ve geçmiş bilgilerini tutar:
```json
{
  "last_update": "2025-11-27 02:00:00",      // Son otomatik güncelleme
  "next_scheduled": "2025-11-28 02:00:00",   // Sonraki planlanan çalışma
  "status": "idle",                          // idle | running | error
  "stocks": {
    "GARAN.IS": {
      "last_data_update": "2025-11-27 02:05:00",
      "data_status": "updated",
      "last_date": "2025-11-26",
      "last_model_update": "2025-11-20 03:15:00",
      "model_name": "LassoLarsCV",
      "r2_score": 0.9410,
      "model_status": "good"
    }
  }
}
```

---

## 🎯 Desteklenen Hisseler

### 🇹🇷 BIST-30 (10 hisse)
```
GARAN.IS    - Garanti Bankası
THYAO.IS    - Türk Hava Yolları
AKBNK.IS    - Akbank
EREGL.IS    - Ereğli Demir Çelik
TUPRS.IS    - Tüpraş
KCHOL.IS    - Koç Holding
SAHOL.IS    - Sabancı Holding
ASELS.IS    - Aselsan
SISE.IS     - Şişe Cam
TCELL.IS    - Turkcell
```

### 🇺🇸 S&P 500 (10 hisse)
```
AAPL    - Apple Inc.
MSFT    - Microsoft Corp.
GOOGL   - Alphabet Inc.
AMZN    - Amazon.com Inc.
TSLA    - Tesla Inc.
META    - Meta Platforms Inc.
NVDA    - NVIDIA Corp.
JPM     - JPMorgan Chase & Co.
V       - Visa Inc.
WMT     - Walmart Inc.
```

---

## 📊 Model Performansı

### Regression Modelleri (Production Ready)

| Hisse | Model | R² Score | RMSE | MAPE | Dataset |
|-------|-------|----------|------|------|---------|
| **GARAN.IS** | LassoLarsCV | **0.9410** | 0.234 | 2.18% | 5 yıl |
| **AAPL** | Ridge | **0.9385** | 1.245 | 1.89% | 5 yıl |
| **MSFT** | HuberRegressor | **0.9799** | 0.856 | 1.54% | 5 yıl |
| **THYAO.IS** | LinearRegression | **0.8980** | 0.412 | 2.67% | 5 yıl |

> **Not:** R² > 0.90 skoru, modelin varyansın %90'ından fazlasını açıklayabildiğini gösterir.

### Backtest Sonuçları (1 Yıl Simülasyonu)

| Hisse | Toplam Getiri | Sharpe Ratio | Max Drawdown | İşlem Sayısı | Kazanma Oranı |
|-------|---------------|--------------|--------------|--------------|---------------|
| **GARAN.IS** | **+37.68%** 🏆 | 1.12 | -25.29% | 18 | 66.7% |
| **AAPL** | +5.45% | 0.33 | -28.67% | 8 | 75.0% |

**Backtest Parametreleri:**
- Başlangıç sermayesi: $10,000
- İşlem başına yatırım: Sermayenin %95'i
- Komisyon: İşlem başına %0.1
- Sinyal threshold'ları: US hisseleri ±1%, Türk hisseleri ±2%
- Test periyodu: Son 1 yıl (252 işlem günü)

---

## 💻 Kullanım Komutları

### 1️⃣ Dashboard (Ana Uygulama)
```bash
streamlit run app.py
```
**Özellikler:**
- 💰 Mevcut fiyat + Yarın tahmini
- 📈 BUY/SELL/HOLD sinyali
- 🔧 15+ teknik gösterge grafiği
- 📊 Backtest performans metrikleri
- 🔄 Manuel veri güncelleme ve model eğitimi (pipeline varsa)

### 2️⃣ Veri Toplama
```bash
python main.py
```
Tüm 20 hisse için 5 yıllık veri indir → `data/raw/*.csv`

### 3️⃣ EDA Analizi
```bash
python run_eda.py
```
Kapsamlı görselleştirmeler oluştur

### 4️⃣ Teknik Analiz
```bash
python run_technical_analysis.py
```
Teknik göstergeleri hesapla → `data/technical/*.csv`

### 5️⃣ Model Test (LazyPredict - 40+ model)
```bash
python run_lazy_predict.py
```
Otomatik model keşfi → `outputs/lazy_predict_results.csv`

### 6️⃣ En İyi Modelleri Eğit
```bash
python scripts/outputs/train_best_models.py
```
Regression modellerini eğit → `models/*.pkl`

### 7️⃣ Backtest Simülasyonu
```bash
python scripts/outputs/backtest.py
```
1 yıllık strateji testi → `outputs/backtest_report.txt`

### 8️⃣ LazyPredict Sonuçlarını Analiz Et
```bash
python scripts/outputs/analyze_lazy_results.py
```
Model performans karşılaştırması

---

## 🐳 Docker Deployment

### docker-compose.yml
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
```

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY . .

# Port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Çalıştır
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Deployment Adımları
```bash
# 1. Image build et
docker-compose build

# 2. Container'ı başlat
docker-compose up -d

# 3. Logları kontrol et
docker-compose logs -f

# 4. Container'a bağlan (debug için)
docker-compose exec exchangetracker bash

# 5. Durdur
docker-compose down
```

---

## 🔧 Pipeline Yapılandırması

### config.py - Temel Ayarlar
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

### Pipeline Çalışma Mantığı

```
┌─────────────────────────────────────────────────────┐
│  GÜNLÜK OTOMATİK ÇALIŞTIRMA (02:00)                │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  Data Updater   │
         │  ─────────────  │
         │  • Yahoo Finance│
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
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ State Update    │
         │ ─────────────── │
         │ • JSON dosyası  │
         │ • Timestamp     │
         │ • Performans    │
         └─────────────────┘
```

**Akıllı Yeniden Eğitim Mantığı:**
1. Mevcut model varsa performansını kontrol et
2. R² < 0.85 ise → Yeniden eğit
3. Son eğitimden 7+ gün geçtiyse → Kontrol et
4. Yeni model daha iyiyse → Değiştir

---

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'feat: Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

### Geliştirme Yol Haritası
- [ ] Real-time veri akışı (WebSocket)
- [ ] Daha fazla teknik gösterge (Ichimoku, Fibonacci)
- [ ] Sentiment analizi (Twitter, Reddit)
- [ ] Portfolio optimizasyonu
- [ ] E-posta/SMS bildirimleri
- [ ] Multi-timeframe analizi (1h, 4h, 1d)

---

## 📜 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 👨‍💻 Geliştirici

**Halil Öztekin**
- 🎓 Konya Teknik Üniversitesi - Bilgisayar Mühendisliği
- 📧 Email: [haliloztekin@protonmail.com]
- 🔗 GitHub: [@hoztekin](https://github.com/hoztekin)
- 💼 LinkedIn: [Halil Öztekin](https://www.linkedin.com/in/halil-o-a3a75b233/)

---

## ⚠️ Yasal Uyarı

**DİKKAT:** Bu yazılım yalnızca eğitim ve araştırma amaçlıdır.

- ❌ Finansal tavsiye değildir
- ❌ Yatırım garantisi vermez
- ❌ Gerçek parayla işlem yapmadan önce profesyonel danışman görüşü alın
- ⚠️ Geçmiş performans gelecek getiriyi garanti etmez
- ⚠️ Borsa yatırımları risk içerir, sermaye kaybı yaşayabilirsiniz

**Geliştirici, bu yazılımın kullanımından kaynaklanan herhangi bir finansal kayıptan sorumlu tutulamaz.**

---

## 🙏 Teşekkürler

- [Streamlit](https://streamlit.io/) - Dashboard framework
- [Yahoo Finance](https://finance.yahoo.com/) - Veri kaynağı
- [LazyPredict](https://github.com/shankarpandala/lazypredict) - Otomatik model seçimi
- [Plotly](https://plotly.com/) - İnteraktif grafikler
- [Scikit-learn](https://scikit-learn.org/) - Makine öğrenmesi kütüphanesi

---

<div align="center">

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**

[⬆ Başa Dön](#-exchangetracker---borsa-trend-analizi-ve-tahmin-sistemi)

</div>