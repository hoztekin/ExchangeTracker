"""
Data Updater Module
Yahoo Finance'den veri çekme ve güncelleme
"""
import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from .config import DATA_DIR, DATA_PERIOD, DATA_INTERVAL, LOG_FORMAT, LOG_LEVEL

# Logging setup
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class DataUpdater:
    def __init__(self):
        self.data_dir = DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

    def get_data_file_path(self, ticker: str) -> str:
        """Ticker için CSV dosya yolunu döndür"""
        return os.path.join(self.data_dir, f"{ticker.replace('.', '_')}_data.csv")

    def load_existing_data(self, ticker: str) -> pd.DataFrame:
        """Mevcut veriyi yükle"""
        file_path = self.get_data_file_path(ticker)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                logger.info(f"✓ {ticker} için mevcut veri yüklendi: {len(df)} kayıt")
                return df
            except Exception as e:
                logger.error(f"✗ {ticker} veri yükleme hatası: {e}")
                return None
        return None

    def fetch_new_data(self, ticker: str, start_date: str = None) -> pd.DataFrame:
        """Yahoo Finance'den yeni veri çek"""
        try:
            if start_date:
                # Sadece belirli tarihten sonrasını çek
                data = yf.download(ticker, start=start_date, progress=False)
            else:
                # Tüm veriyi çek
                data = yf.download(ticker, period=DATA_PERIOD, interval=DATA_INTERVAL, progress=False)

            if data.empty:
                logger.warning(f"⚠ {ticker} için veri bulunamadı")
                return None

            logger.info(f"✓ {ticker} için {len(data)} yeni kayıt indirildi")
            return data

        except Exception as e:
            logger.error(f"✗ {ticker} veri çekme hatası: {e}")
            return None

    def update_stock(self, ticker: str) -> dict:
        """
        Belirli bir hisse için veriyi güncelle ve technical indicators ekle
        Returns: dict with status and info
        """
        logger.info(f"📊 {ticker} verisi güncelleniyor...")

        # Dosya adını düzelt
        clean_ticker = ticker.replace('.', '_').replace('^', '').replace('=', '_')

        # Mevcut veriyi yükle
        existing_data = self.load_existing_data(ticker)

        if existing_data is not None and not existing_data.empty:
            # Son tarihten sonrasını çek
            last_date = existing_data.index[-1]
            next_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            new_data = self.fetch_new_data(ticker, start_date=next_date)

            if new_data is not None and not new_data.empty:
                # Yeni veriyi ekle
                combined_data = pd.concat([existing_data, new_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data.sort_index(inplace=True)

                # Technical indicators ekle
                combined_data = self._add_technical_indicators(combined_data)

                # Kaydet
                file_path = self.get_data_file_path(ticker)
                combined_data.to_csv(file_path)

                logger.info(f"✓ {ticker} verisi güncellendi: +{len(new_data)} yeni kayıt")
                return {
                    'status': 'updated',
                    'ticker': ticker,
                    'new_records': len(new_data),
                    'total_records': len(combined_data),
                    'last_date': str(combined_data.index[-1].date()),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.info(f"ℹ {ticker} için yeni veri yok")
                return {
                    'status': 'no_new_data',
                    'ticker': ticker,
                    'last_date': str(existing_data.index[-1].date()),
                    'timestamp': datetime.now().isoformat()
                }
        else:
            # İlk kez veri çek
            new_data = self.fetch_new_data(ticker)

            if new_data is not None and not new_data.empty:
                # Technical indicators ekle
                new_data = self._add_technical_indicators(new_data)

                file_path = self.get_data_file_path(ticker)
                new_data.to_csv(file_path)

                logger.info(f"✓ {ticker} verisi ilk kez indirildi: {len(new_data)} kayıt")
                return {
                    'status': 'initialized',
                    'ticker': ticker,
                    'total_records': len(new_data),
                    'last_date': str(new_data.index[-1].date()),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.error(f"✗ {ticker} verisi indirilemedi")
                return {
                    'status': 'error',
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat()
                }

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basit technical indicators ekle
        Tam technical analysis için src/utils/indicators.py kullanılabilir
        """
        try:
            # SMA
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()

            # EMA
            df['ema_12'] = df['Close'].ewm(span=12).mean()
            df['ema_26'] = df['Close'].ewm(span=26).mean()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr_14'] = true_range.rolling(14).mean()

            # Kolon adlarını küçük harfe çevir (mevcut yapıyla uyumlu)
            df.columns = df.columns.str.lower()

            return df

        except Exception as e:
            logger.error(f"✗ Technical indicators hatası: {e}")
            return df

    def update_all_stocks(self, stock_list: list) -> dict:
        """
        Tüm hisseler için veri güncelle
        Returns: dict with results for all stocks
        """
        results = {}
        logger.info(f"🔄 {len(stock_list)} hisse için güncelleme başlıyor...")

        for ticker in stock_list:
            results[ticker] = self.update_stock(ticker)

        logger.info("✓ Tüm hisseler güncellendi")
        return results


if __name__ == "__main__":
    # Test
    updater = DataUpdater()
    result = updater.update_stock('GARAN.IS')
    print(result)