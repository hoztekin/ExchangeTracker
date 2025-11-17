"""
LazyPredict ile Model Keşfi - Çalıştırma Scripti
8. Hafta: Otomatik Model Seçimi
"""

# ✅ AAPL (S&P 500)
# ✅ MSFT (S&P 500)
# ✅ GARAN_IS (BIST-30)
# ✅ THYAO_IS (BIST-30)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

try:
    from lazypredict.Supervised import LazyClassifier, LazyRegressor

    LAZYPREDICT_AVAILABLE = True
except ImportError:
    LAZYPREDICT_AVAILABLE = False
    print("⚠️  LazyPredict kurulu değil!")


class FixedLazyModelSelector:
    """Düzeltilmiş LazyPredict - Infinity/NaN sorunlarını çözer"""

    def __init__(self, data_dir='data/technical'):
        self.data_dir = Path(data_dir)
        self.results = {}

        if not LAZYPREDICT_AVAILABLE:
            raise ImportError("LazyPredict kurulu değil!")

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

    def clean_data(self, df):
        """
        KRİTİK: Infinity, NaN, ve aşırı değerleri temizle
        """
        print("   🧹 Veri temizleniyor...")

        # 1. Infinity'leri NaN'a çevir
        df = df.replace([np.inf, -np.inf], np.nan)

        # 2. Çok büyük değerleri NaN'a çevir (float64 limiti)
        for col in df.select_dtypes(include=[np.number]).columns:
            df.loc[np.abs(df[col]) > 1e10, col] = np.nan

        # 3. NaN oranını kontrol et
        nan_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        print(f"   📊 NaN oranı: {nan_ratio:.2%}")

        if nan_ratio > 0.5:
            print(f"   ⚠️  Çok fazla NaN var! Veri kalitesi düşük.")

        return df

    def prepare_features(self, df):
        """Feature'ları hazırla ve temizle"""
        base_features = [
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'stochastic_k', 'stochastic_d', 'williams_r',
            'bb_position', 'atr_14', 'mfi_14',
            'sma_20', 'sma_50', 'ema_12', 'ema_26'
        ]

        # Ekstra feature'lar ekle
        df = self._add_extra_features(df)

        extra_features = [
            'price_change_1d', 'price_change_5d', 'price_change_20d',
            'volume_change_1d', 'momentum_5', 'momentum_10',
            'volatility_5', 'volatility_20'
        ]

        all_features = base_features + extra_features
        available_features = [f for f in all_features if f in df.columns]

        # Veriyi temizle
        df = self.clean_data(df)

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
            # Volume 0 olanları NaN yap
            df.loc[df['volume'] == 0, 'volume_change_1d'] = np.nan

        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)

        # Volatilite
        df['volatility_5'] = df['close'].rolling(5).std()
        df['volatility_20'] = df['close'].rolling(20).std()

        # KRİTİK: pct_change sonrası infinity temizliği
        for col in df.columns:
            if 'change' in col:
                # %1000'den fazla değişimleri NaN yap (muhtemelen hata)
                df.loc[np.abs(df[col]) > 10, col] = np.nan

        return df

    def create_classification_target(self, df, threshold=0.02):
        """Classification hedef - temizlenmiş"""
        df['next_day_return'] = df['close'].shift(-1) / df['close'] - 1

        # Infinity temizle
        df['next_day_return'] = df['next_day_return'].replace([np.inf, -np.inf], np.nan)

        # Aşırı değerleri temizle
        df.loc[np.abs(df['next_day_return']) > 1, 'next_day_return'] = np.nan

        df['target'] = 0  # HOLD
        df.loc[df['next_day_return'] > threshold, 'target'] = 1  # BUY
        df.loc[df['next_day_return'] < -threshold, 'target'] = -1  # SELL

        return df['target']

    def create_regression_target(self, df):
        """Regression hedef"""
        return df['close'].shift(-1)

    def run_classification(self, ticker, threshold=0.02, test_size=0.2):
        """Classification - düzeltilmiş"""
        print(f"\n{'=' * 70}")
        print(f"🔍 {ticker} - CLASSIFICATION (Fixed)")
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

        print(f"   ✅ Temizleme sonrası: {len(X)} satır")

        if len(X) < 100:
            print(f"   ❌ Çok az veri kaldı! ({len(X)} satır)")
            return None

        # 4. Train-test split (TIME-BASED)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\n📊 Dataset:")
        print(f"   • Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"   • Features: {X.shape[1]}")

        # 5. Normalize - clip extreme values
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # KRİTİK: Scaled veriden de infinity kontrolü
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=10.0, neginf=-10.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

        print(f"\n🚀 LazyPredict çalışıyor...\n")

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

            print("\n" + "=" * 70)
            print("📊 EN İYİ 10 MODEL")
            print("=" * 70)
            print(models.sort_values('F1 Score', ascending=False).head(10).to_string())
            print("=" * 70 + "\n")

            self.results[f'{ticker}_classification'] = {
                'models': models,
                'predictions': predictions,
                'scaler': scaler,
                'features': X.columns.tolist()
            }

            return models

        except Exception as e:
            print(f"\n❌ Hata: {str(e)}")
            return None

    def run_regression(self, ticker, test_size=0.2):
        """Regression - düzeltilmiş"""
        print(f"\n{'=' * 70}")
        print(f"📈 {ticker} - REGRESSION (Fixed)")
        print(f"{'=' * 70}\n")

        df = self.load_technical_data(ticker)
        if df is None:
            return None

        print(f"   ✅ Veri yüklendi: {len(df)} satır")

        X = self.prepare_features(df)
        y = self.create_regression_target(df)

        # NaN temizle
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"   ✅ Temizleme sonrası: {len(X)} satır")

        if len(X) < 100:
            print(f"   ❌ Çok az veri kaldı!")
            return None

        # Train-test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\n📊 Dataset:")
        print(f"   • Train: {len(X_train)}, Test: {len(X_test)}")

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Infinity kontrolü
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=10.0, neginf=-10.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

        print(f"\n🚀 LazyPredict çalışıyor...\n")

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

            print("\n" + "=" * 70)
            print("📊 EN İYİ 10 MODEL")
            print("=" * 70)
            print(models.sort_values('R-Squared', ascending=False).head(10).to_string())
            print("=" * 70 + "\n")

            self.results[f'{ticker}_regression'] = {
                'models': models,
                'predictions': predictions,
                'scaler': scaler,
                'features': X.columns.tolist()
            }

            return models

        except Exception as e:
            print(f"\n❌ Hata: {str(e)}")
            return None

    def save_results(self, output_dir='outputs/lazy_predict'):
        """Sonuçları kaydet"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"💾 SONUÇLARI KAYDETME")
        print(f"{'=' * 70}\n")

        for key, result in self.results.items():
            filepath = output_dir / f"{key}_results.csv"
            result['models'].to_csv(filepath)
            print(f"✅ {filepath}")

        print(f"\n📁 Kaydedildi: {output_dir}\n")


def main():
    """Test"""
    print("=" * 70)
    print("🔧 FIXED LAZYPREDICT - Infinity/NaN Temizleyici")
    print("=" * 70 + "\n")

    selector = FixedLazyModelSelector()

    # Önce BIST hissesi test et
    test_tickers = ['THYAO_IS', 'AAPL']

    for ticker in test_tickers:
        print(f"\n{'🔹' * 35}")
        print(f"🎯 {ticker} TEST EDİLİYOR")
        print(f"{'🔹' * 35}\n")

        # Classification
        clf_result = selector.run_classification(ticker, threshold=0.02)

        if clf_result is not None:
            input("\n▶️  Regression için ENTER...")

            # Regression
            reg_result = selector.run_regression(ticker)

        if ticker != test_tickers[-1]:
            input(f"\n▶️  Sonraki hisse ({test_tickers[test_tickers.index(ticker) + 1]}) için ENTER...")

    # Kaydet
    selector.save_results()

    print("\n" + "=" * 70)
    print("✨ TAMAMLANDI!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Durduruldu.")
    except Exception as e:
        print(f"\n❌ Hata: {str(e)}")
        import traceback

        traceback.print_exc()