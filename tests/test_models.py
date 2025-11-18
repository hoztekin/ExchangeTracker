"""
Model Test Scripti
Kaydedilmiş modelleri yükler, test eder ve tahmin yapar
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class ModelTester:
    """Kaydedilmiş modelleri test et"""

    def __init__(self, models_dir='models', data_dir='data/technical'):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.loaded_models = {}

    def load_model(self, ticker, model_name):
        """
        Modeli ve ilgili dosyaları yükle

        Returns:
        - dict: model, scaler, features, metadata
        """
        clean_ticker = ticker.replace('.', '_').replace('^', '').replace('=', '_')

        model_file = self.models_dir / f"{clean_ticker}_{model_name}_model.pkl"
        scaler_file = self.models_dir / f"{clean_ticker}_{model_name}_scaler.pkl"
        features_file = self.models_dir / f"{clean_ticker}_{model_name}_features.pkl"
        metadata_file = self.models_dir / f"{clean_ticker}_{model_name}_metadata.pkl"

        # Kontrol
        if not model_file.exists():
            print(f"❌ Model dosyası bulunamadı: {model_file}")
            return None

        # Yükle
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        features = joblib.load(features_file)
        metadata = joblib.load(metadata_file)

        key = f"{ticker}_{model_name}"
        self.loaded_models[key] = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metadata': metadata,
            'ticker': ticker,
            'model_name': model_name
        }

        return self.loaded_models[key]

    def load_all_models(self):
        """Tüm kaydedilmiş modelleri yükle"""
        print("📂 Modeller yükleniyor...\n")

        # models/ klasöründeki tüm _model.pkl dosyalarını bul
        model_files = list(self.models_dir.glob('*_model.pkl'))

        if not model_files:
            print(f"❌ '{self.models_dir}' klasöründe model bulunamadı!")
            return

        for model_file in model_files:
            # Dosya adından ticker ve model_name çıkar
            # Örnek: AAPL_ridge_model.pkl
            filename = model_file.stem.replace('_model', '')
            parts = filename.split('_')

            # Son kısım model adı, geri kalanı ticker
            model_name = parts[-1]
            ticker = '_'.join(parts[:-1])

            result = self.load_model(ticker, model_name)
            if result:
                print(f"✅ {ticker:15s} - {model_name:10s} → Yüklendi")

        print(f"\n📊 Toplam {len(self.loaded_models)} model yüklendi\n")

    def load_recent_data(self, ticker, days=30):
        """Son N günün verilerini yükle"""
        filename = ticker.replace('.', '_').replace('^', '').replace('=', '_')
        filepath = self.data_dir / f"{filename}_technical.csv"

        if not filepath.exists():
            return None

        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Son N gün
        return df.tail(days)

    def prepare_features(self, df, feature_list):
        """Model için feature'ları hazırla"""
        # Eksik feature'ları hesapla
        if 'price_change_1d' not in df.columns:
            df['price_change_1d'] = df['close'].pct_change()
        if 'price_change_5d' not in df.columns:
            df['price_change_5d'] = df['close'].pct_change(5)
        if 'momentum_10' not in df.columns:
            df['momentum_10'] = df['close'] - df['close'].shift(10)
        if 'volatility_20' not in df.columns:
            df['volatility_20'] = df['close'].rolling(20).std()
        if 'volume' in df.columns and 'volume_ratio' not in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20'].replace(0, np.nan)

        # Temizlik
        df = df.replace([np.inf, -np.inf], np.nan)

        # Sadece gerekli feature'ları al
        return df[feature_list]

    def predict(self, ticker, model_name, date=None):
        """
        Belirli bir tarih için tahmin yap

        Parameters:
        - ticker: Hisse kodu
        - model_name: Model adı
        - date: Tahmin tarihi (None ise en son tarih)

        Returns:
        - dict: tahmin, gerçek, hata
        """
        key = f"{ticker}_{model_name}"

        if key not in self.loaded_models:
            print(f"❌ Model yüklü değil: {key}")
            return None

        model_data = self.loaded_models[key]
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']

        # Veri yükle
        df = self.load_recent_data(ticker, days=50)
        if df is None:
            print(f"❌ Veri bulunamadı: {ticker}")
            return None

        # Feature hazırla
        X = self.prepare_features(df, features)

        # Gerçek değer (yarının fiyatı)
        y_true = df['close'].shift(-1)

        # NaN temizle
        valid_idx = X.notna().all(axis=1) & y_true.notna()
        X = X[valid_idx]
        y_true = y_true[valid_idx]

        if len(X) == 0:
            print(f"❌ Geçerli veri yok")
            return None

        # Tarih seç
        if date is None:
            # En son geçerli tarih
            idx = -1
            pred_date = X.index[idx]
            actual_price = df.loc[pred_date, 'close']
        else:
            # Belirli tarih
            if date not in X.index:
                print(f"❌ Tarih bulunamadı: {date}")
                return None
            pred_date = date
            actual_price = df.loc[pred_date, 'close']

        # Feature al
        X_pred = X.loc[pred_date:pred_date]

        # Scale
        X_scaled = scaler.transform(X_pred)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

        # Tahmin
        y_pred = model.predict(X_scaled)[0]
        y_actual = y_true.loc[pred_date]

        # Hata
        error = y_pred - y_actual
        error_pct = (error / y_actual) * 100 if y_actual != 0 else 0

        return {
            'date': pred_date,
            'today_price': actual_price,
            'predicted': y_pred,
            'actual': y_actual,
            'error': error,
            'error_pct': error_pct,
            'ticker': ticker,
            'model_name': model_name
        }

    def test_model(self, ticker, model_name, test_days=10):
        """
        Son N gün için modeli test et

        Returns:
        - DataFrame: test sonuçları
        """
        print(f"\n{'=' * 70}")
        print(f"🧪 {ticker} - {model_name.upper()} TEST")
        print(f"{'=' * 70}\n")

        key = f"{ticker}_{model_name}"

        if key not in self.loaded_models:
            print(f"❌ Model yüklü değil: {key}")
            return None

        model_data = self.loaded_models[key]
        metadata = model_data['metadata']

        # Metadata göster
        print(f"📊 Model Bilgileri:")
        print(f"   Test R²:  {metadata['test_r2']:.4f}")
        print(f"   RMSE:     {metadata['rmse']:.3f}")
        print(f"   MAPE:     {metadata['mape']:.2f}%")

        # Veri yükle
        df = self.load_recent_data(ticker, days=50)
        if df is None:
            return None

        features = model_data['features']
        X = self.prepare_features(df, features)
        y_true = df['close'].shift(-1)

        # Temizle
        valid_idx = X.notna().all(axis=1) & y_true.notna()
        X = X[valid_idx]
        y_true = y_true[valid_idx]
        df_filtered = df[valid_idx]

        # Son N gün
        X_test = X.tail(test_days)
        y_test = y_true.tail(test_days)
        df_test = df_filtered.tail(test_days)

        # Scale
        scaler = model_data['scaler']
        X_scaled = scaler.transform(X_test)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

        # Tahmin
        model = model_data['model']
        y_pred = model.predict(X_scaled)

        # Sonuçlar
        results = []
        for i in range(len(X_test)):
            date = X_test.index[i]
            today = df_test.loc[date, 'close']
            pred = y_pred[i]
            actual = y_test.iloc[i]
            error = pred - actual
            error_pct = (error / actual) * 100 if actual != 0 else 0

            # Sinyal
            change_pct = ((pred - today) / today) * 100
            if change_pct > 2:
                signal = "BUY 📈"
            elif change_pct < -2:
                signal = "SELL 📉"
            else:
                signal = "HOLD ⏸️"

            results.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Today': today,
                'Predicted': pred,
                'Actual': actual,
                'Error': error,
                'Error (%)': error_pct,
                'Signal': signal
            })

        df_results = pd.DataFrame(results)

        # Göster
        print(f"\n📊 Son {test_days} Gün Test Sonuçları:")
        print(f"{'─' * 70}")
        print(df_results.to_string(index=False))

        # İstatistikler
        print(f"\n📈 Test İstatistikleri:")
        print(f"   Ortalama Hata:     {df_results['Error'].mean():.3f}")
        print(f"   Ortalama Hata (%): {df_results['Error (%)'].abs().mean():.2f}%")
        print(f"   Max Hata:          {df_results['Error'].abs().max():.3f}")
        print(f"   RMSE:              {np.sqrt((df_results['Error'] ** 2).mean()):.3f}")

        return df_results

    def predict_tomorrow(self, ticker, model_name):
        """
        Yarının fiyatını tahmin et

        Returns:
        - dict: tahmin bilgileri
        """
        print(f"\n{'=' * 70}")
        print(f"🔮 YARIN TAHMİNİ: {ticker} - {model_name.upper()}")
        print(f"{'=' * 70}\n")

        key = f"{ticker}_{model_name}"

        if key not in self.loaded_models:
            print(f"❌ Model yüklü değil: {key}")
            return None

        model_data = self.loaded_models[key]

        # Son veriyi al
        df = self.load_recent_data(ticker, days=50)
        if df is None:
            return None

        # En son fiyat
        latest_date = df.index[-1]
        latest_price = df['close'].iloc[-1]

        print(f"📅 Bugün: {latest_date.strftime('%Y-%m-%d')}")
        print(f"💰 Bugünkü Fiyat: {latest_price:.2f}")

        # Feature hazırla
        features = model_data['features']
        X = self.prepare_features(df, features)

        # En son satır
        X_latest = X.iloc[-1:].copy()

        # NaN kontrolü
        if X_latest.isna().any().any():
            print(f"⚠️  Eksik veriler var, tahmin yapılamıyor")
            return None

        # Scale
        scaler = model_data['scaler']
        X_scaled = scaler.transform(X_latest)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

        # Tahmin
        model = model_data['model']
        tomorrow_pred = model.predict(X_scaled)[0]

        # Değişim
        change = tomorrow_pred - latest_price
        change_pct = (change / latest_price) * 100

        # Sinyal
        if change_pct > 2:
            signal = "BUY 📈"
            signal_emoji = "🟢"
        elif change_pct < -2:
            signal = "SELL 📉"
            signal_emoji = "🔴"
        else:
            signal = "HOLD ⏸️"
            signal_emoji = "🟡"

        print(f"\n🔮 Yarın Tahmini: {tomorrow_pred:.2f}")
        print(f"📊 Değişim: {change:+.2f} ({change_pct:+.2f}%)")
        print(f"{signal_emoji} Sinyal: {signal}")

        # Güven aralığı (basit yaklaşım: ±RMSE)
        metadata = model_data['metadata']
        rmse = metadata['rmse']

        print(f"\n📊 Güven Aralığı (±RMSE):")
        print(f"   Alt Sınır:  {tomorrow_pred - rmse:.2f}")
        print(f"   Tahmin:     {tomorrow_pred:.2f}")
        print(f"   Üst Sınır:  {tomorrow_pred + rmse:.2f}")

        return {
            'ticker': ticker,
            'model_name': model_name,
            'today_date': latest_date,
            'today_price': latest_price,
            'tomorrow_pred': tomorrow_pred,
            'change': change,
            'change_pct': change_pct,
            'signal': signal,
            'confidence_lower': tomorrow_pred - rmse,
            'confidence_upper': tomorrow_pred + rmse,
            'model_r2': metadata['test_r2'],
            'model_mape': metadata['mape']
        }

    def test_all_models(self):
        """Tüm yüklü modelleri test et"""
        print("\n" + "=" * 70)
        print("🧪 TÜM MODELLER TEST EDİLİYOR")
        print("=" * 70)

        all_predictions = []

        for key, model_data in self.loaded_models.items():
            ticker = model_data['ticker']
            model_name = model_data['model_name']

            pred = self.predict_tomorrow(ticker, model_name)
            if pred:
                all_predictions.append(pred)

            input("\n▶️  Sonraki model için ENTER...")

        # Özet tablo
        if all_predictions:
            print("\n" + "=" * 70)
            print("📊 YARIN TAHMİNLERİ ÖZETİ")
            print("=" * 70 + "\n")

            df_summary = pd.DataFrame(all_predictions)
            df_summary = df_summary[['ticker', 'today_price', 'tomorrow_pred',
                                     'change_pct', 'signal', 'model_r2']]

            print(df_summary.to_string(index=False))
            print("\n" + "=" * 70 + "\n")


def main():
    """Ana program"""

    print("=" * 70)
    print("🧪 MODEL TEST SİSTEMİ")
    print("=" * 70)
    print("\nBu script:")
    print("  1. Kaydedilmiş modelleri yükler")
    print("  2. Test verileriyle performansı ölçer")
    print("  3. Yarının fiyatını tahmin eder")
    print("  4. BUY/SELL/HOLD sinyali üretir")
    print("=" * 70)

    input("\n▶️  Başlamak için ENTER...")

    tester = ModelTester()

    # 1. Modelleri yükle
    tester.load_all_models()

    if not tester.loaded_models:
        print("\n❌ Model bulunamadı!")
        return

    input("\n▶️  Test etmeye başlamak için ENTER...")

    # 2. Tüm modelleri test et
    tester.test_all_models()

    print("\n" + "=" * 70)
    print("✅ TÜM TESTLER TAMAMLANDI!")
    print("=" * 70)
    print("\n🎯 Sonraki adım: Backtesting (Kar/Zarar Simülasyonu)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem durduruldu.")
    except Exception as e:
        print(f"\n❌ Hata: {str(e)}")
        import traceback

        traceback.print_exc()