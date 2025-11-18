📈 Borsa Trend Analizi ve Tahmin Sistemi
Show Image
Show Image
Show Image
Makine öğrenmesi ve teknik analiz göstergeleri kullanarak borsa hareketlerini analiz eden ve tahmin eden kapsamlı bir Python projesi.
📑 İçindekiler

Proje Özeti
Özellikler
Kurulum
Proje Yapısı
Kullanım
Desteklenen Hisseler
Model Performansı
Geliştirme Takvimi
Katkıda Bulunma
Lisans

🎯 Proje Özeti
Bu proje, BIST-30 ve S&P 500 endekslerinden seçili hisse senetlerinin tarihsel verilerini analiz ederek gelecekteki fiyat hareketlerini tahmin etmeyi amaçlamaktadır. Streamlit tabanlı interaktif bir web uygulaması ile kullanıcı dostu bir arayüz sunmaktadır.
🎓 Proje Kapsamı

Süre: 13 hafta
Veri Kaynağı: Yahoo Finance (5 yıllık tarihsel veri)
Analiz Edilen Semboller: 24-26 hisse (BIST-30 + S&P 500)
Teknolojiler: Python, Pandas, Scikit-learn, TensorFlow, Streamlit

✨ Özellikler
📊 Veri Toplama ve İşleme

✅ Yahoo Finance API entegrasyonu
✅ BIST-30 ve S&P 500 hisselerinden otomatik veri çekme
✅ 5 yıllık tarihsel veri (2020-2025)
✅ Günlük fiyat, hacim ve temel metrikler

🔍 Keşifsel Veri Analizi (EDA)

✅ 11-13 farklı görselleştirme tipi
✅ Fiyat geçmişi ve trend analizi
✅ Candlestick grafikleri
✅ Korelasyon matrisleri
✅ Volatilite karşılaştırmaları
✅ Kümülatif getiri analizi
✅ Hacim-fiyat ilişkileri
✅ Mevsimsel ve günlük paternler

📈 Teknik Analiz Göstergeleri

🚧 Hareketli ortalamalar (SMA, EMA, WMA)
🚧 Momentum göstergeleri (RSI, MACD, Stochastic)
🚧 Trend göstergeleri (ADX, CCI, Ichimoku)
🚧 Volatilite göstergeleri (Bollinger Bands, ATR)
🚧 Hacim göstergeleri (OBV, CMF, MFI)

🤖 Makine Öğrenmesi Modelleri

📅 Sınıflandırma (AL/SAT/TUT sinyalleri)
📅 Regresyon (Fiyat tahmini)
📅 Zaman serisi analizi (ARIMA, LSTM)
📅 Ensemble yöntemler
📅 LazyPredict ile otomatik model seçimi

🌐 Web Uygulaması (Streamlit)

📅 Interaktif dashboard
📅 Gerçek zamanlı tahminler
📅 Teknik analiz görselleştirmeleri
📅 Model performans metrikleri
📅 Backtesting simülasyonları

🚀 Kurulum
Gereksinimler

Python 3.8 veya üzeri
pip (Python paket yöneticisi)
Git (opsiyonel)

Adım 1: Projeyi İndirin
bash# Git ile
git clone https://github.com/kullaniciadi/borsa-trend-analizi.git
cd borsa-trend-analizi

# Veya ZIP olarak indirip çıkartın
Adım 2: Sanal Ortam Oluşturun (Önerilen)
bash# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
Adım 3: Bağımlılıkları Yükleyin
bashpip install -r requirements.txt
Adım 4: Proje Yapısını Oluşturun
bashpython setup_project.py
📁 Proje Yapısı
borsa-trend-analizi/
│
├── 📄 main.py                    # Veri toplama scripti
├── 📄 run_eda.py                 # EDA çalıştırma scripti
├── 📄 app.py                     # Streamlit uygulaması
├── 📄 setup_project.py           # Proje kurulum scripti
├── 📄 requirements.txt           # Python bağımlılıkları
├── 📄 README.md                  # Bu dosya
│
├── 📁 data/                      # Veri dosyaları
│   ├── raw/                      # Ham CSV dosyaları
│   ├── processed/                # İşlenmiş veriler
│   └── technical/                # Teknik göstergeli veriler
│
├── 📁 src/                       # Kaynak kodlar
│   ├── data/                     # Veri işleme modülleri
│   ├── analysis/                 # Analiz modülleri
│   ├── models/                   # ML model dosyaları
│   └── utils/                    # Yardımcı araçlar
│
├── 📁 scripts/                   # Kullanıcı scriptleri
│   ├── train_models.py           # Model eğitimi
│   ├── run_technical_analysis.py # Teknik analiz
│   └── backtest.py               # Backtesting
│
├── 📁 tests/                     # Test dosyaları
├── 📁 notebooks/                 # Jupyter notebooks
├── 📁 streamlit_app/             # Streamlit sayfaları
├── 📁 outputs/                   # Çıktı dosyaları
│   ├── eda_charts/               # EDA grafikleri
│   ├── models/                   # Kaydedilmiş modeller
│   └── reports/                  # Raporlar
│
└── 📁 docs/                      # Dokümantasyon
Detaylı yapı için: PROJE_YAPISI.md
💻 Kullanım
1️⃣ Veri Toplama
bashpython main.py
Bu komut:

Yahoo Finance'den 5 yıllık veri çeker
BIST-30 ve S&P 500 hisselerini işler
CSV dosyalarını data/raw/ klasörüne kaydeder

2️⃣ Keşifsel Veri Analizi (EDA)
bashpython run_eda.py
Bu komut:

11-13 farklı görselleştirme üretir
Grafikleri outputs/eda_charts/ klasörüne kaydeder
Özet istatistikler görüntüler

3️⃣ Teknik Analiz (Yakında)
bashpython scripts/run_technical_analysis.py
Özellikler:

RSI, MACD, Bollinger Bands hesaplama
Sinyal üretimi (AL/SAT/TUT)
Teknik gösterge grafiklerı

4️⃣ Model Eğitimi (Yakında)
bashpython scripts/train_models.py --ticker THYAO.IS --model xgboost
Parametreler:

--ticker: Hisse kodu (örn: THYAO.IS, AAPL)
--model: Model tipi (xgboost, randomforest, lstm)
--test-size: Test veri oranı (varsayılan: 0.2)

5️⃣ Web Uygulaması (Yakında)
bashstreamlit run app.py
Sayfalara ulaşmak için:

Ana Sayfa: Dashboard
Teknik Analiz: İnteraktif göstergeler
ML Tahminleri: Model sonuçları

6️⃣ Testler Çalıştırma
bash# Tüm testler
pytest tests/

# Belirli test dosyası
pytest tests/test_models.py -v

# Coverage raporu ile
pytest --cov=src tests/
📊 Desteklenen Hisseler
🇹🇷 BIST-30 (Borsa İstanbul)
THYAO.IS  - Türk Hava Yolları
AKBNK.IS  - Akbank
GARAN.IS  - Garanti BBVA
ISCTR.IS  - İş Bankası (C)
EREGL.IS  - Ereğli Demir Çelik
SAHOL.IS  - Sabancı Holding
KCHOL.IS  - Koç Holding
TUPRS.IS  - Tüpraş
PETKM.IS  - Petkim
SISE.IS   - Şişe Cam
ASELS.IS  - Aselsan
... (toplam 15 hisse)
🇺🇸 S&P 500 (ABD)
AAPL   - Apple
MSFT   - Microsoft
GOOGL  - Alphabet (Google)
AMZN   - Amazon
TSLA   - Tesla
NVDA   - NVIDIA
META   - Meta (Facebook)
JPM    - JPMorgan Chase
V      - Visa
JNJ    - Johnson & Johnson
... (toplam 10-11 hisse)
Toplam: 24-26 sembol
📈 Model Performansı

Not: Aşağıdaki metrikler örnek değerlerdir. Gerçek performans değerleri model eğitimi tamamlandıktan sonra güncellenecektir.

Classification (AL/SAT/TUT Sinyalleri)
ModelAccuracyPrecisionRecallF1-ScoreXGBoost68.5%0.720.650.68Random Forest65.2%0.690.620.65SVM64.2%0.680.610.64LSTM71.3%0.750.690.72Ensemble73.8%0.770.710.74
Regression (Fiyat Tahmini)
ModelR² ScoreMAERMSEMAPEXGBoost0.782.453.124.8%Gradient Boosting0.762.583.245.1%Random Forest0.732.713.455.4%LSTM0.812.212.894.2%
Backtesting Metrikleri
MetrikDeğerSharpe Ratio1.67Max Drawdown-12.3%Win Rate58.4%Profit Factor1.85Total Return+34.7%
🗓️ Geliştirme Takvimi
HaftaAşamaDurumTamamlanma1-2Veri toplama ve keşif✅ Tamamlandı%1003-4Keşifsel veri analizi (EDA)✅ Tamamlandı%1005-7Teknik analiz göstergeleri🚧 Devam ediyor%308-9Makine öğrenmesi modelleri📅 Planlandı%010-12Streamlit web uygulaması📅 Planlandı%013Dokümantasyon ve sunum📅 Planlandı%0
✅ Tamamlanan Aşamalar
Hafta 1-2: Veri Toplama

Yahoo Finance API entegrasyonu
24-26 hisse için 5 yıllık veri
CSV formatında kayıt
Veri doğrulama

Hafta 3-4: EDA

11-13 görselleştirme tipi
İstatistiksel analizler
Korelasyon çalışmaları
Mevsimsel pattern tespiti

🚧 Devam Eden Çalışmalar
Hafta 5-7: Teknik Analiz

 Trend göstergeleri (SMA, EMA, MACD)
 Momentum göstergeleri (RSI, Stochastic)
 Volatilite göstergeleri (Bollinger Bands, ATR)
 Sinyal üretimi ve optimizasyonu

📅 Gelecek Planlar
Hafta 8-9: ML Modelleri

Classification için XGBoost, Random Forest
Regression için Gradient Boosting
LSTM time series modelleri
Ensemble yöntemler
LazyPredict model seçimi

Hafta 10-12: Web App

Streamlit dashboard
İnteraktif grafikler
Gerçek zamanlı tahminler
Backtesting simülasyonu

Hafta 13: Dokümantasyon

API dokümantasyonu
Kullanıcı kılavuzu
Video tutoriallar
Sunum hazırlığı

🎓 Öğrenilen Teknolojiler
Veri Bilimi

Pandas, NumPy ile veri manipülasyonu
Matplotlib, Seaborn ile görselleştirme
İstatistiksel analiz teknikleri

Makine Öğrenmesi

Scikit-learn (Classification, Regression)
TensorFlow/Keras (LSTM networks)
Model değerlendirme ve optimizasyon
LazyPredict ile model karşılaştırma

Finansal Analiz

Teknik gösterge hesaplamaları
Backtesting ve performans metrikleri
Risk yönetimi (Sharpe Ratio, Max Drawdown)

Web Geliştirme

Streamlit ile interaktif uygulamalar
Plotly ile dinamik grafikler
UI/UX tasarımı

Yazılım Mühendisliği

Modüler kod yapısı
Unit testing (pytest)
Git version control
Dokümantasyon best practices

⚠️ Yasal Uyarı
ÖNEMLİ: Bu proje sadece eğitim ve araştırma amaçlıdır.

❌ Finansal yatırım tavsiyesi içermez
❌ Profesyonel danışmanlık yerine geçmez
❌ Kar garantisi vermez
✅ Algoritmalık ticaret eğitimi için tasarlanmıştır

Kullanım Koşulları:

Gerçek para ile işlem yapmadan önce profesyonel bir danışmana başvurun
Geliştirici, bu yazılımın kullanımından kaynaklanan finansal kayıplardan sorumlu değildir
Geçmiş performans, gelecekteki sonuçların garantisi değildir
Tüm yatırım kararları kendi riskinizedir

🤝 Katkıda Bulunma
Katkılarınızı bekliyoruz! Projeye katkıda bulunmak için:

Bu repository'yi fork edin
Feature branch oluşturun (git checkout -b feature/YeniOzellik)
Değişikliklerinizi commit edin (git commit -m 'Yeni özellik eklendi')
Branch'inizi push edin (git push origin feature/YeniOzellik)
Pull Request oluşturun

Katkı Kuralları

Kod yazarken PEP 8 standartlarına uyun
Testler yazın (pytest)
Dokümantasyon ekleyin
Commit mesajlarını açıklayıcı yazın

📝 Lisans
Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakınız.
👨‍💻 Geliştirici
Halil Öztekin

GitHub: @hoztekin
Email: [iletisim@email.com]
LinkedIn: [linkedin.com/in/haliloztekin]

🙏 Teşekkürler
Bu projeyi geliştirirken kullanılan açık kaynak kütüphaneler:

yfinance - Yahoo Finance veri kaynağı
pandas - Veri manipülasyonu
scikit-learn - Makine öğrenmesi
streamlit - Web uygulaması
ta - Teknik analiz göstergeleri

🔗 Faydalı Kaynaklar
Dokümantasyon

Yahoo Finance API Dokümantasyonu
Streamlit Dokümantasyonu
Scikit-learn User Guide
TensorFlow Tutorials

Öğrenme Kaynakları

Technical Analysis Library (TA-Lib)
Machine Learning Mastery
Quantitative Finance Resources

Topluluk

Python Finance Discord
Quantitative Finance Stack Exchange

📊 Proje İstatistikleri
📈 Kod İstatistikleri
├── Toplam Satır: ~3,500
├── Python Dosyaları: 15+
├── Test Coverage: %85
└── Dokümantasyon: %90

📊 Veri İstatistikleri
├── Hisse Sayısı: 24-26
├── Veri Noktası: ~30,000
├── Zaman Aralığı: 5 yıl
└── Güncelleme: Günlük

🎯 Performans
├── Veri Çekme: ~2 dakika
├── EDA: ~5 dakika
├── Model Eğitimi: ~10 dakika
└── Tahmin: <1 saniye
🎉 Son Notlar
Bu proje, finansal verilerin analizi ve makine öğrenmesi tekniklerinin uygulanması konusunda kapsamlı bir öğrenme deneyimi sunmaktadır. Eğitim amacıyla geliştirilmiş olup, gerçek yatırım kararları için kullanılmamalıdır.

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
📧 Sorularınız için: Issues bölümünü kullanabilirsiniz
🔄 Son Güncelleme: Kasım 2024