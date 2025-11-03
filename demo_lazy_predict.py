"""
Borsa Trend Analizi - LazyPredict Model Seçici
8. Hafta: Otomatik Model Keşfi
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# LazyPredict'i conditional olarak import et
try:
    from lazypredict.Supervised import LazyClassifier, LazyRegressor

    LAZYPREDICT_AVAILABLE = True
except ImportError:
    LAZYPREDICT_AVAILABLE = False
    print("⚠️  LazyPredict kurulu değil! 'pip install lazypredict' çalıştırın.")


class LazyModelSelector:
    """LazyPredict kullanarak en iyi modelleri bul"""

    def __init__(self, data_dir='data/technical'):
        """
        Parameters:
        - data_dir: Teknik analiz CSV dosyalarının bulunduğu klasör
        """
        self.data_dir = Path(data_dir)
        self.data = {}
        self.results = {}

        if not LAZYPREDICT_AVAILABLE:
            raise ImportError("LazyPredict kurulu değil! 'pip install lazypredict' çalıştırın.")

    def load_technical_data(self, ticker):
        """Teknik analiz verileri yükle"""
        filename = ticker.replace('^', '').replace('=', '_').replace('.', '_')
        filepath = self.data_dir / f"{filename}_technical.csv"

        if filepath.exists():
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
        else:
            print(f"⚠️  {filepath} bulunamadı!")
            return None

    def prepare_features(self, df):
        """
        Feature'ları hazırla

        Returns:
        - DataFrame with selected features
        """
        # Temel teknik göstergeler
        base_features = [
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'stochastic_k', 'stochastic_d', 'williams_r',
            'bb_position', 'atr_14', 'mfi_14',
            'sma_20', 'sma_50', 'ema_12', 'ema_26'
        ]

        # Ekstra feature'lar ekle
        df = self._add_extra_features(df)

        # Ekstra feature listesi
        extra_features = [
            'price_change_1d', 'price_change_5d', 'price_change_20d',
            'volume_change_1d', 'momentum_5', 'momentum_10',
            'volatility_5', 'volatility_20'
        ]

        all_features = base_features + extra_features

        # Sadece var olan kolonları al
        available_features = [f for f in all_features if f in df.columns]

        print(f"   📊 Kullanılan feature sayısı: {len(available_features)}")

        return df[available_features]

    def _add_extra_features(self, df):
        """Ekstra feature'lar ekle"""
        # Fiyat değişimleri
        df['price_change_1d'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_20d'] = df['close'].pct_change(20)

        # Volume değişimi
        if 'volume' in df.columns:
            df['volume_change_1d'] = df['volume'].pct_change()

        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)

        # Volatilite (farklı periyotlar)
        df['volatility_5'] = df['close'].rolling(5).std()
        df['volatility_20'] = df['close'].rolling(20).std()

        return df

    def create_classification_target(self, df, threshold=0.02):
        """
        Classification hedef değişkeni oluştur

        Parameters:
        - df: DataFrame
        - threshold: Alım/satım eşiği (default: %2)

        Returns:
        - Series with target values: BUY=1, HOLD=0, SELL=-1
        """
        # Yarının getirisi
        df['next_day_return'] = df['close'].shift(-1) / df['close'] - 1

        # Sinyal oluştur
        df['target'] = 0  # HOLD
        df.loc[df['next_day_return'] > threshold, 'target'] = 1  # BUY
        df.loc[df['next_day_return'] < -threshold, 'target'] = -1  # SELL

        return df['target']

    def create_regression_target(self, df):
        """
        Regression hedef değişkeni oluştur

        Returns:
        - Series: Yarının kapanış fiyatı
        """
        return df['close'].shift(-1)

    def run_classification(self, ticker, threshold=0.02, test_size=0.2):
        """
        Classification için LazyPredict çalıştır

        Parameters:
        - ticker: Hisse kodu
        - threshold: Alım/satım eşiği
        - test_size: Test set oranı

        Returns:
        - DataFrame: Model karşılaştırma sonuçları
        """
        print(f"\n{'=' * 70}")
        print(f"🔍 {ticker} - CLASSIFICATION MODEL DISCOVERY")
        print(f"{'=' * 70}\n")

        # 1. Veriyi yükle
        df = self.load_technical_data(ticker)
        if df is None:
            return None

        print(f"   ✅ Veri yüklendi: {len(df)} satır")

        # 2. Feature'ları hazırla
        X = self.prepare_features(df)
        y = self.create_classification_target(df, threshold)

        # 3. NaN'ları temizle
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"   ✅ Temizleme sonrası: {len(X)} satır\n")

        # 4. Train-test split (TIME-BASED!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"📊 Dataset Bilgileri:")
        print(f"   • Train size: {len(X_train)}")
        print(f"   • Test size: {len(X_test)}")
        print(f"   • Feature count: {X.shape[1]}")

        print(f"\n📊 Class Distribution (Train):")
        print(f"   • BUY (1):  {(y_train == 1).sum():4d} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
        print(f"   • HOLD (0): {(y_train == 0).sum():4d} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
        print(f"   • SELL (-1): {(y_train == -1).sum():4d} ({(y_train == -1).sum() / len(y_train) * 100:.1f}%)\n")

        # 5. Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 6. LazyPredict
        print("🚀 LazyPredict çalışıyor (bu biraz zaman alabilir)...")
        print("   📝 40+ model test ediliyor...\n")

        try:
            clf = LazyClassifier(
                verbose=0,
                ignore_warnings=True,
                custom_metric=None,
                predictions=True
            )

            models, predictions = clf.fit(
                X_train_scaled, X_test_scaled,
                y_train, y_test
            )

            # 7. Sonuçları göster
            print("\n" + "=" * 70)
            print("📊 EN İYİ 10 CLASSIFICATION MODEL")
            print("=" * 70)
            print(models.sort_values('Accuracy', ascending=False).head(10).to_string())
            print("=" * 70 + "\n")

            # 8. Kaydet
            self.results[f'{ticker}_classification'] = {
                'models': models,
                'predictions': predictions,
                'scaler': scaler,
                'features': X.columns.tolist(),
                'test_size': test_size,
                'threshold': threshold
            }

            return models

        except Exception as e:
            print(f"\n❌ LazyPredict hatası: {str(e)}")
            print(f"   Muhtemel neden: Bazı modellerde sorun var")
            return None

    def run_regression(self, ticker, test_size=0.2):
        """
        Regression için LazyPredict çalıştır

        Parameters:
        - ticker: Hisse kodu
        - test_size: Test set oranı

        Returns:
        - DataFrame: Model karşılaştırma sonuçları
        """
        print(f"\n{'=' * 70}")
        print(f"📈 {ticker} - REGRESSION MODEL DISCOVERY")
        print(f"{'=' * 70}\n")

        # 1. Veriyi yükle
        df = self.load_technical_data(ticker)
        if df is None:
            return None

        print(f"   ✅ Veri yüklendi: {len(df)} satır")

        # 2. Feature'ları hazırla
        X = self.prepare_features(df)
        y = self.create_regression_target(df)

        # 3. NaN'ları temizle
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"   ✅ Temizleme sonrası: {len(X)} satır\n")

        # 4. Train-test split (TIME-BASED!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"📊 Dataset Bilgileri:")
        print(f"   • Train size: {len(X_train)}")
        print(f"   • Test size: {len(X_test)}")
        print(f"   • Feature count: {X.shape[1]}")
        print(f"   • Target range: {y_train.min():.2f} - {y_train.max():.2f}\n")

        # 5. Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 6. LazyPredict
        print("🚀 LazyPredict çalışıyor (bu biraz zaman alabilir)...")
        print("   📝 40+ model test ediliyor...\n")

        try:
            reg = LazyRegressor(
                verbose=0,
                ignore_warnings=True,
                predictions=True
            )

            models, predictions = reg.fit(
                X_train_scaled, X_test_scaled,
                y_train, y_test
            )

            # 7. Sonuçları göster
            print("\n" + "=" * 70)
            print("📊 EN İYİ 10 REGRESSION MODEL")
            print("=" * 70)
            print(models.sort_values('R-Squared', ascending=False).head(10).to_string())
            print("=" * 70 + "\n")

            # 8. Kaydet
            self.results[f'{ticker}_regression'] = {
                'models': models,
                'predictions': predictions,
                'scaler': scaler,
                'features': X.columns.tolist(),
                'test_size': test_size
            }

            return models

        except Exception as e:
            print(f"\n❌ LazyPredict hatası: {str(e)}")
            print(f"   Muhtemel neden: Bazı modellerde sorun var")
            return None

    def get_top_models(self, ticker, task='classification', n=5):
        """
        En iyi N modeli döndür

        Parameters:
        - ticker: Hisse kodu
        - task: 'classification' veya 'regression'
        - n: Döndürülecek model sayısı

        Returns:
        - DataFrame: En iyi N model
        """
        result_key = f'{ticker}_{task}'

        if result_key not in self.results:
            print(f"❌ {result_key} sonucu bulunamadı!")
            return None

        models_df = self.results[result_key]['models']

        if task == 'classification':
            return models_df.sort_values('Accuracy', ascending=False).head(n)
        else:
            return models_df.sort_values('R-Squared', ascending=False).head(n)

    def save_results(self, output_dir='outputs/lazy_predict'):
        """
        Sonuçları CSV olarak kaydet

        Parameters:
        - output_dir: Çıktı klasörü
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"💾 SONUÇLARI KAYDETME")
        print(f"{'=' * 70}\n")

        for key, result in self.results.items():
            # Model karşılaştırma tablosunu kaydet
            filepath = output_dir / f"{key}_results.csv"
            result['models'].to_csv(filepath)
            print(f"✅ {filepath}")

        print(f"\n📁 Tüm sonuçlar kaydedildi: {output_dir}")
        print(f"={'=' * 70}\n")

    def generate_summary_report(self, output_dir='outputs/lazy_predict'):
        """
        Tüm sonuçlar için özet rapor oluştur

        Parameters:
        - output_dir: Çıktı klasörü
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / 'summary_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("📊 LAZYPREDICT - MODEL DISCOVERY ÖZET RAPORU\n")
            f.write("=" * 70 + "\n\n")

            for key, result in self.results.items():
                f.write(f"\n{'=' * 70}\n")
                f.write(f"📊 {key.upper()}\n")
                f.write(f"{'=' * 70}\n\n")

                models = result['models']

                if 'classification' in key:
                    top_5 = models.sort_values('Accuracy', ascending=False).head(5)
                    f.write("EN İYİ 5 MODEL (Accuracy):\n\n")
                    f.write(top_5[['Accuracy', 'Balanced Accuracy', 'F1 Score', 'Time Taken']].to_string())
                else:
                    top_5 = models.sort_values('R-Squared', ascending=False).head(5)
                    f.write("EN İYİ 5 MODEL (R-Squared):\n\n")
                    f.write(top_5[['R-Squared', 'RMSE', 'MAE', 'Time Taken']].to_string())

                f.write("\n\n")
                f.write(f"Feature Count: {len(result['features'])}\n")
                f.write(f"Test Size: {result.get('test_size', 'N/A')}\n")

                if 'threshold' in result:
                    f.write(f"Classification Threshold: ±{result['threshold'] * 100}%\n")

                f.write("\n")

        print(f"\n✅ Özet rapor oluşturuldu: {report_path}\n")


if __name__ == "__main__":
    print("📊 LazyPredict Model Selector hazır!")
    print("\nKullanım:")
    print("  from src.models.lazy_model_selector import LazyModelSelector")
    print("  selector = LazyModelSelector()")
    print("  selector.run_classification('THYAO_IS')")
    print("  selector.run_regression('AAPL')")