"""
Model Trainer Module
Model eğitimi, değerlendirme ve kaydetme
"""
import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from lazypredict.Supervised import LazyRegressor
from .config import MODELS_DIR, DATA_DIR, MIN_R2_SCORE, LOG_FORMAT, LOG_LEVEL

# Logging setup
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.data_dir = DATA_DIR
        os.makedirs(self.models_dir, exist_ok=True)

    def get_model_file_path(self, ticker: str) -> str:
        """Ticker için model dosya yolunu döndür"""
        return os.path.join(self.models_dir, f"{ticker.replace('.', '_')}_model.pkl")

    def load_data(self, ticker: str) -> pd.DataFrame:
        """CSV'den veri yükle - technical klasöründen"""
        # Ticker formatını düzelt
        filename = ticker.replace('.', '_').replace('^', '').replace('=', '_')
        file_path = os.path.join(self.data_dir, f"{filename}_technical.csv")

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Date kolonu varsa index yap
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                logger.info(f"✓ {ticker} verisi yüklendi: {len(df)} kayıt")
                return df
            except Exception as e:
                logger.error(f"✗ {ticker} veri yükleme hatası: {e}")
                return None
        else:
            logger.error(f"✗ {ticker} veri dosyası bulunamadı: {file_path}")
            return None

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Feature engineering ve train/test split
        Mevcut technical analysis verilerini kullanır
        """
        try:
            # NaN ve Inf değerlerini temizle
            df = df.replace([np.inf, -np.inf], np.nan)

            # Temel teknik göstergeler (mevcut kolonlar)
            base_features = [
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                'stochastic_k', 'stochastic_d', 'williams_r',
                'bb_position', 'atr_14', 'mfi_14',
                'sma_20', 'sma_50', 'ema_12', 'ema_26'
            ]

            # Ekstra feature'lar ekle
            if 'price_change_1d' not in df.columns:
                df['price_change_1d'] = df['close'].pct_change()
            if 'price_change_5d' not in df.columns:
                df['price_change_5d'] = df['close'].pct_change(5)
            if 'momentum_10' not in df.columns:
                df['momentum_10'] = df['close'] - df['close'].shift(10)
            if 'volatility_20' not in df.columns:
                df['volatility_20'] = df['close'].rolling(20).std()

            extra_features = [
                'price_change_1d', 'price_change_5d',
                'momentum_10', 'volatility_20'
            ]

            # Sadece var olan feature'ları kullan
            all_features = base_features + extra_features
            available_features = [f for f in all_features if f in df.columns]

            # Target: Next day's close
            df['target'] = df['close'].shift(-1)

            # NaN temizliği
            df = df.dropna()

            # Features ve target ayır
            X = df[available_features]
            y = df['target']

            # Train/test split (time-based, %80-%20)
            split_idx = int(len(df) * 0.8)
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]

            logger.info(f"✓ Features hazırlandı: {len(X_train)} train, {len(X_test)} test")
            logger.info(f"✓ Feature sayısı: {len(available_features)}")
            return X_train, X_test, y_train, y_test, available_features

        except Exception as e:
            logger.error(f"✗ Feature hazırlama hatası: {e}")
            return None, None, None, None, None

    def train_model(self, ticker: str, force_retrain: bool = False) -> dict:
        """
        Model eğit veya mevcut modeli değerlendir
        """
        logger.info(f"🤖 {ticker} için model eğitimi başlıyor...")

        # Veriyi yükle
        df = self.load_data(ticker)
        if df is None:
            return {'status': 'error', 'message': 'Veri yüklenemedi'}

        # Features hazırla
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_features(df)
        if X_train is None:
            return {'status': 'error', 'message': 'Feature hazırlama hatası'}

        # Mevcut model var mı kontrol et
        model_path = self.get_model_file_path(ticker)
        retrain_needed = force_retrain

        if not force_retrain and os.path.exists(model_path):
            # Mevcut modeli yükle ve değerlendir
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                model = model_data['model']
                y_pred = model.predict(X_test)
                current_r2 = r2_score(y_test, y_pred)

                logger.info(f"ℹ Mevcut model R²: {current_r2:.4f}")

                if current_r2 < MIN_R2_SCORE:
                    logger.warning(f"⚠ R² score düşük ({current_r2:.4f} < {MIN_R2_SCORE}), yeniden eğitim gerekli")
                    retrain_needed = True
                else:
                    logger.info(f"✓ Mevcut model yeterli ({current_r2:.4f} >= {MIN_R2_SCORE})")
                    return {
                        'status': 'existing_model_good',
                        'ticker': ticker,
                        'model_name': model_data.get('model_name', 'Unknown'),
                        'r2_score': current_r2,
                        'last_trained': model_data.get('trained_date', 'Unknown'),
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                logger.error(f"✗ Model yükleme hatası: {e}, yeniden eğitim yapılacak")
                retrain_needed = True

        # Model eğitimi gerekiyorsa
        if retrain_needed or not os.path.exists(model_path):
            try:
                logger.info("🔄 LazyPredict ile model eğitimi başlıyor...")

                # LazyPredict
                reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
                models, predictions = reg.fit(X_train, X_test, y_train, y_test)

                # En iyi modeli seç
                best_model_name = models.index[0]
                best_r2 = models.iloc[0]['R-Squared']

                logger.info(f"✓ En iyi model: {best_model_name} (R²: {best_r2:.4f})")

                # MAPE hesapla
                y_pred = predictions[best_model_name]
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100

                # Model datasını kaydet
                model_data = {
                    'model': reg.models[best_model_name],
                    'model_name': best_model_name,
                    'r2_score': best_r2,
                    'mape': mape,
                    'feature_cols': feature_cols,
                    'trained_date': datetime.now().isoformat(),
                    'data_range': f"{df.index[0].date()} to {df.index[-1].date()}",
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                }

                # Modeli kaydet
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)

                logger.info(f"✓ Model kaydedildi: {model_path}")

                return {
                    'status': 'trained',
                    'ticker': ticker,
                    'model_name': best_model_name,
                    'r2_score': best_r2,
                    'mape': mape,
                    'trained_date': model_data['trained_date'],
                    'timestamp': datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"✗ Model eğitim hatası: {e}")
                return {
                    'status': 'error',
                    'ticker': ticker,
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }

    def evaluate_existing_model(self, ticker: str) -> dict:
        """Mevcut modelin performansını değerlendir"""
        model_path = self.get_model_file_path(ticker)

        if not os.path.exists(model_path):
            return {'status': 'no_model', 'message': 'Model bulunamadı'}

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Güncel veri ile test et
            df = self.load_data(ticker)
            if df is None:
                return {'status': 'error', 'message': 'Veri yüklenemedi'}

            X_train, X_test, y_train, y_test, _ = self.prepare_features(df)
            if X_test is None:
                return {'status': 'error', 'message': 'Feature hazırlama hatası'}

            model = model_data['model']
            y_pred = model.predict(X_test)
            current_r2 = r2_score(y_test, y_pred)
            current_mape = mean_absolute_percentage_error(y_test, y_pred) * 100

            return {
                'status': 'evaluated',
                'ticker': ticker,
                'model_name': model_data.get('model_name', 'Unknown'),
                'r2_score': current_r2,
                'mape': current_mape,
                'trained_date': model_data.get('trained_date', 'Unknown'),
                'needs_retraining': current_r2 < MIN_R2_SCORE,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"✗ Model değerlendirme hatası: {e}")
            return {'status': 'error', 'message': str(e)}


if __name__ == "__main__":
    # Test
    trainer = ModelTrainer()
    result = trainer.train_model('GARAN.IS')
    print(result)