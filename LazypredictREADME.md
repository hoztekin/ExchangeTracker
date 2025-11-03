# 🚀 LazyPredict - Otomatik Model Keşfi

## 📋 Nedir?

LazyPredict, tek komutla 40+ makine öğrenmesi modelini otomatik olarak test edip karşılaştıran bir kütüphanedir. Bu sayede hangi modellerin projeniz için en iyi çalıştığını hızlıca keşfedebilirsiniz.

## 🎯 Ne İşe Yarar?

- ✅ **Hızlı Model Keşfi**: Hangi model ailesi işe yarar?
- ✅ **Zaman Tasarrufu**: 40 model tek komutla test edilir
- ✅ **Objektif Karşılaştırma**: Accuracy, R², RMSE vb. metriklerle
- ✅ **Baseline Belirleme**: Hangi modellerle devam edeceğinize karar verin

## 📦 Kurulum

```bash
# Tüm gerekli kütüphaneleri yükle
pip install -r requirements.txt

# Veya sadece LazyPredict için:
pip install lazypredict xgboost lightgbm catboost
```

## 🚀 Hızlı Başlangıç

### 1. Demo ile Test Et (Önerilen!)

```bash
python demo_lazy_predict.py
```

Bu script:
- Demo verisi oluşturur
- Classification test eder (sinyal tahmini)
- Regression test eder (fiyat tahmini)
- Sonuçları `outputs/lazy_predict_demo/` klasörüne kaydeder

### 2. Gerçek Verilerle Çalıştır

```bash
python run_lazy_predict.py
```

Bu script:
- `data/technical/` klasöründeki tüm hisseleri test eder
- Her hisse için Classification ve Regression çalıştırır
- Sonuçları `outputs/lazy_predict/` klasörüne kaydeder

## 📊 Çıktılar

### Classification Sonuçları

```
Model                          Accuracy  Balanced Accuracy  F1 Score  Time
XGBClassifier                      0.73              0.71      0.72   2.1s
LGBMClassifier                     0.72              0.70      0.71   1.8s
RandomForestClassifier             0.71              0.69      0.70   3.2s
```

**Metrikler:**
- **Accuracy**: Doğru tahmin oranı
- **Balanced Accuracy**: Class imbalance düzeltilmiş accuracy
- **F1 Score**: Precision ve recall dengesi
- **Time**: Eğitim süresi

### Regression Sonuçları

```
Model                          R-Squared   RMSE     MAE    Time
XGBRegressor                       0.87    2.45    1.82    2.3s
LGBMRegressor                      0.86    2.51    1.89    1.9s
RandomForestRegressor              0.85    2.58    1.95    3.5s
```

**Metrikler:**
- **R-Squared**: Model açıklama gücü (0-1 arası, yüksek iyi)
- **RMSE**: Root Mean Squared Error (düşük iyi)
- **MAE**: Mean Absolute Error (düşük iyi)
- **Time**: Eğitim süresi

## 🎯 Sonuçları Nasıl Yorumlarız?

### En İyi Modelleri Seç

1. **Accuracy/R² En Yüksek**: Tahmin başarısı en iyi
2. **Time Dengesi**: Çok yavaş modeller production'da sorun olabilir
3. **Consistency**: Farklı hisselerde benzer performans gösterenler tercih edilir

### Tipik Kazananlar

**Classification için:**
- 🥇 XGBoost
- 🥈 LightGBM
- 🥉 RandomForest

**Regression için:**
- 🥇 XGBoost
- 🥈 GradientBoosting
- 🥉 ExtraTrees

## 🔧 Kod Kullanımı

```python
from src.models.lazy_model_selector import LazyModelSelector

# 1. Selector'ı başlat
selector = LazyModelSelector(data_dir='data/technical')

# 2. Classification test et
clf_results = selector.run_classification(
    ticker='THYAO_IS',
    threshold=0.02,  # ±%2 eşik (BUY/SELL/HOLD)
    test_size=0.2    # Test set %20
)

# 3. Regression test et
reg_results = selector.run_regression(
    ticker='AAPL',
    test_size=0.2
)

# 4. En iyi modelleri al
top_5_clf = selector.get_top_models('THYAO_IS', task='classification', n=5)
top_5_reg = selector.get_top_models('AAPL', task='regression', n=5)

# 5. Sonuçları kaydet
selector.save_results(output_dir='outputs/lazy_predict')
selector.generate_summary_report()
```

## 📂 Klasör Yapısı

```
outputs/lazy_predict/
├── THYAO_IS_classification_results.csv    # Classification sonuçları
├── THYAO_IS_regression_results.csv        # Regression sonuçları
├── AAPL_classification_results.csv
├── AAPL_regression_results.csv
└── summary_report.txt                      # Özet rapor
```

## ⚠️ Önemli Notlar

### LazyPredict'in Sınırlamaları

❌ **Hiperparametre optimizasyonu YOK**
- Default ayarlarla test eder
- En iyi modelleri seçtikten sonra GridSearchCV ile optimize edin

❌ **Walk-forward validation YOK**
- Basit train-test split kullanır
- Time series için özel validation yapın

❌ **Finansal metrikler YOK**
- Sharpe Ratio, Max Drawdown hesaplanmaz
- Backtesting ayrıca yapılmalı

❌ **Ensemble/Stacking YOK**
- Modelleri birleştirme yapmaz
- Voting/Stacking ayrıca kodlanmalı

### İyi Pratikler

✅ **2 Aşamalı Yaklaşım:**
1. **Aşama 1**: LazyPredict ile hızlı tarama
2. **Aşama 2**: En iyi 3-5 modeli derinlemesine optimize et

✅ **Time-Based Split:**
- Script otomatik olarak time-based split kullanır
- Random shuffle YAPILMAZ (zaman serisi için önemli!)

✅ **Feature Scaling:**
- StandardScaler otomatik uygulanır
- Tüm modeller normalize edilmiş veri görür

## 🎯 Sonraki Adımlar

LazyPredict sonuçlarına göre:

### 1. Model Seçimi
- En iyi 3-5 modeli belirle
- Farklı hisselerde benzer performans gösterenleri seç

### 2. Hiperparametre Tuning
```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid = GridSearchCV(xgb.XGBClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
```

### 3. Walk-Forward Validation
- Her gün yeniden eğit
- Gerçekçi performans ölç

### 4. Backtesting
- Trading simülasyonu yap
- Sharpe Ratio, Max Drawdown hesapla

### 5. Production
- En iyi modeli kaydet (.pkl)
- Streamlit'e entegre et

## 💡 Örnek Workflow

```bash
# 1. Demo ile test
python demo_lazy_predict.py

# 2. Gerçek verilerle çalıştır
python run_lazy_predict.py

# 3. Sonuçları incele
cat outputs/lazy_predict/summary_report.txt

# 4. En iyi modelleri seç
# → XGBoost, LightGBM, RandomForest

# 5. Optimize et (bir sonraki adım)
python run_hyperparameter_tuning.py
```

## 🆘 Sorun Giderme

### Hata: "LazyPredict kurulu değil"
```bash
pip install lazypredict
```

### Hata: "Data dosyası bulunamadı"
```bash
# Önce teknik analiz çalıştırın:
python run_technical.py
```

### Hata: "Some models failed"
- Normal! Bazı modeller her veri setinde çalışmayabilir
- LazyPredict otomatik olarak atlar
- Başarılı modellere odaklanın

## 📚 Referanslar

- [LazyPredict Dokümantasyonu](https://github.com/shankarpandala/lazypredict)
- [XGBoost](https://xgboost.readthedocs.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/)

---

🎉 **Artık hazırsın!** LazyPredict ile en iyi modelleri keşfet ve projeni bir üst seviyeye taşı!