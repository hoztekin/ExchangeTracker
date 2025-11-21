"""
En İyi Regression Modellerini Eğit ve Kaydet
LazyPredict sonuçlarına göre production'a hazır modeller
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import joblib
import warnings

warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LassoLarsCV

    LASSOLARS_AVAILABLE = True
except ImportError:
    LASSOLARS_AVAILABLE = False
    print("⚠️  LassoLarsCV import hatası")


class BestModelTrainer:
    """En iyi regression modellerini eğit ve kaydet"""

    def __init__(self, data_dir='data/technical'):
        self.data_dir = Path(data_dir)
        self.trained_models = {}

    def load_data(self, ticker):
        """Teknik analiz verilerini yükle"""
        filename = ticker.replace('.', '_').replace('^', '').replace('=', '_')
        filepath = self.data_dir / f"{filename}_technical.csv"

        if filepath.exists():
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
        else:
            print(f"❌ {filepath} bulunamadı!")
            return None

    def prepare_features(self, df):
        """Feature'ları hazırla"""
        # Base features - LazyPredict'te kullanılanlar
        features = [
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'stochastic_k', 'stochastic_d', 'williams_r',
            'bb_position', 'atr_14', 'mfi_14',
            'sma_20', 'sma_50', 'ema_12', 'ema_26'
        ]

        # Ekstra features (basit ve etkili)
        df['price_change_1d'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['volatility_20'] = df['close'].rolling(20).std()

        features.extend(['price_change_1d', 'price_change_5d',
                         'momentum_10', 'volatility_20'])

        # Volume (eğer varsa)
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20'].replace(0, np.nan)
            features.append('volume_ratio')

        # Temizlik
        df = df.replace([np.inf, -np.inf], np.nan)

        # Outlier kontrolü
        for col in features:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = df[col].clip(lower=mean - 5 * std, upper=mean + 5 * std)

        available = [f for f in features if f in df.columns]
        return df[available]

    def create_target(self, df):
        """Regression target: Yarının kapanış fiyatı"""
        return df['close'].shift(-1)

    def train_and_save(self, ticker, model_name, model, test_size=0.2):
        """
        Model eğit, değerlendir ve kaydet

        Parameters:
        - ticker: Hisse kodu
        - model_name: Model adı (dosya için)
        - model: Model instance
        - test_size: Test set oranı
        """
        print(f"\n{'=' * 70}")
        print(f"🔧 {ticker} - {model_name.upper()}")
        print(f"{'=' * 70}\n")

        # Veri yükle
        df = self.load_data(ticker)
        if df is None:
            return None

        print(f"   📂 Veri yüklendi: {len(df)} satır")

        # Feature hazırla
        X = self.prepare_features(df)
        y = self.create_target(df)

        # Temizle
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"   🧹 Temizleme sonrası: {len(X)} satır, {X.shape[1]} feature")

        if len(X) < 100:
            print(f"   ❌ Yetersiz veri!")
            return None

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # NaN ve Inf kontrolü
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=10.0, neginf=-10.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

        print(f"\n   🚀 Model eğitiliyor...")

        # Model eğit
        try:
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"   ❌ Eğitim hatası: {str(e)}")
            return None

        # Performans
        train_r2 = model.score(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)

        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae = np.mean(np.abs(y_test - y_pred))

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        print(f"\n   📊 PERFORMANS:")
        print(f"      ✅ Train R²:  {train_r2:.4f}")
        print(f"      ✅ Test R²:   {test_r2:.4f}")
        print(f"      📉 RMSE:      {rmse:.3f}")
        print(f"      📉 MAE:       {mae:.3f}")
        print(f"      📉 MAPE:      {mape:.2f}%")

        # Overfitting kontrolü
        if train_r2 - test_r2 > 0.1:
            print(f"      ⚠️  Overfitting olabilir (fark: {train_r2 - test_r2:.3f})")

        # Cross-validation (opsiyonel, yavaş)
        print(f"\n   🔄 Cross-validation yapılıyor...")
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                    cv=tscv, scoring='r2')
        print(f"      ✅ CV R² Ortalama: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # Modeli kaydet
        model_dir = Path('../models')
        model_dir.mkdir(exist_ok=True)

        clean_ticker = ticker.replace('.', '_').replace('^', '').replace('=', '_')

        model_file = model_dir / f"{clean_ticker}_{model_name}_model.pkl"
        scaler_file = model_dir / f"{clean_ticker}_{model_name}_scaler.pkl"
        features_file = model_dir / f"{clean_ticker}_{model_name}_features.pkl"
        metadata_file = model_dir / f"{clean_ticker}_{model_name}_metadata.pkl"

        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)
        joblib.dump(X.columns.tolist(), features_file)

        # Metadata
        metadata = {
            'ticker': ticker,
            'model_name': model_name,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'features': X.columns.tolist(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_period': f"{X_train.index[0]} to {X_train.index[-1]}",
            'test_period': f"{X_test.index[0]} to {X_test.index[-1]}"
        }
        joblib.dump(metadata, metadata_file)

        print(f"\n   💾 KAYDEDILDI:")
        print(f"      • Model:    {model_file.name}")
        print(f"      • Scaler:   {scaler_file.name}")
        print(f"      • Features: {features_file.name}")
        print(f"      • Metadata: {metadata_file.name}")

        # Kaydet
        key = f"{ticker}_{model_name}"
        self.trained_models[key] = metadata

        return metadata

    def generate_report(self, output_file='outputs/trained_models_report.txt'):
        """Eğitim raporu oluştur"""

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("📊 REGRESSION MODELLER EĞİTİM RAPORU\n")
            f.write("=" * 70 + "\n\n")

            # Özet tablo
            f.write("GENEL ÖZET:\n")
            f.write("─" * 70 + "\n")
            f.write(f"{'Model':<30} {'Test R²':>10} {'RMSE':>10} {'MAPE':>10}\n")
            f.write("─" * 70 + "\n")

            for key, meta in self.trained_models.items():
                model_label = f"{meta['ticker']} - {meta['model_name']}"
                f.write(f"{model_label:<30} {meta['test_r2']:>10.4f} "
                        f"{meta['rmse']:>10.3f} {meta['mape']:>9.2f}%\n")

            f.write("─" * 70 + "\n\n")

            # Detaylı bilgiler
            for key, meta in self.trained_models.items():
                f.write("=" * 70 + "\n")
                f.write(f"📌 {meta['ticker']} - {meta['model_name'].upper()}\n")
                f.write("=" * 70 + "\n\n")

                f.write("PERFORMANS METRİKLERİ:\n")
                f.write(f"   Train R²:        {meta['train_r2']:.4f}\n")
                f.write(f"   Test R²:         {meta['test_r2']:.4f}\n")
                f.write(f"   RMSE:            {meta['rmse']:.3f}\n")
                f.write(f"   MAE:             {meta['mae']:.3f}\n")
                f.write(f"   MAPE:            {meta['mape']:.2f}%\n")
                f.write(f"   CV R² (Mean):    {meta['cv_mean']:.4f}\n")
                f.write(f"   CV R² (Std):     {meta['cv_std']:.4f}\n\n")

                f.write("VERİ BİLGİSİ:\n")
                f.write(f"   Train Samples:   {meta['train_samples']}\n")
                f.write(f"   Test Samples:    {meta['test_samples']}\n")
                f.write(f"   Features:        {len(meta['features'])}\n")
                f.write(f"   Train Period:    {meta['train_period']}\n")
                f.write(f"   Test Period:     {meta['test_period']}\n\n")

                # Performans değerlendirmesi
                if meta['test_r2'] >= 0.90:
                    grade = "🏆 MÜKEMMEL"
                elif meta['test_r2'] >= 0.80:
                    grade = "✅ ÇOK İYİ"
                elif meta['test_r2'] >= 0.70:
                    grade = "⚠️  ORTA"
                else:
                    grade = "❌ ZAYIF"

                f.write(f"DEĞERLENDİRME: {grade}\n")
                if meta['test_r2'] >= 0.80:
                    f.write("   ✅ Model production'a hazır!\n")
                else:
                    f.write("   ⚠️  Model iyileştirme gerektirebilir\n")

                f.write("\n")

        print(f"\n✅ Detaylı rapor kaydedildi: {output_file}\n")

    def print_summary(self):
        """Terminal'e özet yazdır"""
        print("\n" + "=" * 70)
        print("📊 EĞİTİM ÖZETİ")
        print("=" * 70 + "\n")

        summary_data = []
        for key, meta in self.trained_models.items():
            summary_data.append({
                'Model': f"{meta['ticker']} - {meta['model_name']}",
                'Test R²': meta['test_r2'],
                'RMSE': meta['rmse'],
                'MAPE (%)': meta['mape']
            })

        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))

        print("\n" + "=" * 70)
        avg_r2 = np.mean([m['test_r2'] for m in self.trained_models.values()])
        print(f"📊 Ortalama Test R²: {avg_r2:.4f}")

        if avg_r2 >= 0.90:
            print("🏆 TÜM MODELLER MÜKEMMEL PERFORMANS!")
        elif avg_r2 >= 0.80:
            print("✅ Modeller çok iyi performans gösteriyor")

        print("=" * 70 + "\n")


def main():
    """Ana program"""

    print("=" * 70)
    print("🏆 EN İYİ REGRESSION MODELLERİNİ EĞİT VE KAYDET")
    print("=" * 70)
    print("\nLazyPredict sonuçlarına göre en iyi modeller:")
    print("\n📊 Eğitilecek Modeller:")
    print("   1. AAPL  → Ridge           (R² = 0.9385)")
    print("   2. MSFT  → HuberRegressor  (R² = 0.9799)")
    print("   3. GARAN → LassoLarsCV     (R² = 0.9410)")
    print("   4. THYAO → LinearRegression (R² = 0.8980)")
    print("\n" + "=" * 70)

    input("\n▶️  Başlamak için ENTER'a basın...")

    trainer = BestModelTrainer()

    # 1. AAPL - Ridge
    trainer.train_and_save(
        ticker='AAPL',
        model_name='ridge',
        model=Ridge(alpha=1.0, random_state=42)
    )

    input("\n▶️  Sonraki model için ENTER...")

    # 2. MSFT - HuberRegressor
    trainer.train_and_save(
        ticker='MSFT',
        model_name='huber',
        model=HuberRegressor(max_iter=200)
    )

    input("\n▶️  Sonraki model için ENTER...")

    # 3. GARAN - LassoLarsCV
    if LASSOLARS_AVAILABLE:
        trainer.train_and_save(
            ticker='GARAN_IS',
            model_name='lassolars',
            model=LassoLarsCV(cv=3, max_iter=500)
        )
    else:
        # Alternatif: Ridge
        print("   ⚠️  LassoLarsCV kullanılamıyor, Ridge kullanılıyor...")
        trainer.train_and_save(
            ticker='GARAN_IS',
            model_name='ridge',
            model=Ridge(alpha=1.0, random_state=42)
        )

    input("\n▶️  Sonraki model için ENTER...")

    # 4. THYAO - LinearRegression
    trainer.train_and_save(
        ticker='THYAO_IS',
        model_name='linear',
        model=LinearRegression()
    )

    # Özet ve rapor
    trainer.print_summary()
    trainer.generate_report()

    print("\n" + "=" * 70)
    print("✅ TÜM MODELLER KAYDEDİLDİ!")
    print("=" * 70)
    print("\n📁 Dosyalar:")
    print("   • models/           → Model .pkl dosyaları")
    print("   • outputs/          → Detaylı rapor")
    print("\n🎯 Sonraki Adımlar:")
    print("   1. Modelleri yükle ve test et")
    print("   2. Backtesting yap")
    print("   3. Streamlit'e entegre et")
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