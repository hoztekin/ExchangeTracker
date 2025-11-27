"""
Data Updater Module - GERÇEK FIX
Yahoo Finance'den veri çekme ve güncelleme
"""
import yfinance as yf
import pandas as pd
import numpy as np
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
        """Mevcut veriyi yükle - FIXED VERSION"""
        file_path = self.get_data_file_path(ticker)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0)
                # CRITICAL FIX: errors='coerce' ekle, duplicate temizle
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[~df.index.duplicated(keep='last')]
                df = df[df.index.notna()]
                df.sort_index(inplace=True)
                logger.info(f"OK {ticker} icin mevcut veri yuklendi: {len(df)} kayit")
                return df
            except Exception as e:
                logger.error(f"HATA {ticker} veri yukleme hatasi: {e}")
                return None
        return None

    def fetch_new_data(self, ticker: str, start_date: str = None) -> pd.DataFrame:
        """Yahoo Finance'den yeni veri çek"""
        try:
            yahoo_ticker = ticker.replace('_', '.')

            if start_date:
                data = yf.download(yahoo_ticker, start=start_date, progress=False)
            else:
                data = yf.download(yahoo_ticker, period=DATA_PERIOD, interval=DATA_INTERVAL, progress=False)

            if data.empty:
                logger.warning(f"WARNING {yahoo_ticker} icin veri bulunamadi")
                return None

            logger.info(f"OK {yahoo_ticker} icin {len(data)} yeni kayit indirildi")
            return data

        except Exception as e:
            logger.error(f"HATA {ticker} veri cekme hatasi: {e}")
            return None

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔥 CRITICAL FIX: MultiIndex kolonları düzgün normalize et

        yfinance şöyle döndürür:
        [('Close', 'GARAN.IS'), ('Volume', 'GARAN.IS'), ('Open', 'GARAN.IS')]

        ESKİ KOD (YANLIŞ):
        [col[-1] for col in columns] → ['GARAN.IS', 'GARAN.IS', 'GARAN.IS'] ❌ DUPLICATE!

        YENİ KOD (DOĞRU):
        [col[0] for col in columns] → ['Close', 'Volume', 'Open'] ✓
        """
        if isinstance(df.columns, pd.MultiIndex):
            # İLK elemanı al (Close, Volume, Open vb.)
            df.columns = [col[0].lower() if isinstance(col, tuple) else str(col).lower() for col in df.columns]
        else:
            df.columns = df.columns.str.lower()
        return df

    def update_stock(self, ticker: str) -> dict:
        """Belirli bir hisse için veriyi güncelle"""
        logger.info(f"Guncelleniyor {ticker} verisi...")

        # Ticker normalizasyonu
        if isinstance(ticker, (list, tuple)):
            if not ticker:
                return {'status': 'error', 'ticker': None, 'message': 'Bos ticker listesi'}
            ticker = ticker[0]

        api_ticker = ticker.replace('_', '.')
        storage_ticker = ticker.replace('.', '_').replace('^', '').replace('=', '_')

        # Mevcut veri yükle
        existing_data = self.load_existing_data(storage_ticker)

        if existing_data is not None and not existing_data.empty:
            # DOUBLE CHECK: Index temizliği
            existing_data.index = pd.to_datetime(existing_data.index, errors='coerce')
            existing_data = existing_data[existing_data.index.notna()]
            existing_data = existing_data[~existing_data.index.duplicated(keep='last')]
            existing_data.sort_index(inplace=True)

            # Son tarih
            last_date = existing_data.index[-1]
            next_date_dt = last_date + timedelta(days=1)
            next_date = next_date_dt.strftime('%Y-%m-%d')

            # Bugün kontrolü
            today = datetime.now().date()
            if next_date_dt.date() > today:
                logger.info(f"{storage_ticker} icin yeni seans kapanmadi, veri yok.")
                return {
                    'status': 'no_new_data',
                    'ticker': storage_ticker,
                    'last_date': last_date.date().isoformat(),
                    'timestamp': datetime.now().isoformat()
                }

            new_data = self.fetch_new_data(api_ticker, start_date=next_date)

            if new_data is not None and not new_data.empty:
                # Index normalize
                new_data.index = pd.to_datetime(new_data.index)

                # 🔥 CRITICAL FIX: Kolon normalize (doğru metod)
                new_data = self._normalize_columns(new_data)

                # Birleştir
                combined_data = pd.concat([existing_data, new_data])
                # CRITICAL: Tekrar duplicate temizle!
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data.sort_index(inplace=True)

                combined_data = self._add_technical_indicators(combined_data)

                file_path = self.get_data_file_path(storage_ticker)
                combined_data.to_csv(file_path)

                logger.info(f"OK {storage_ticker} verisi guncellendi: +{len(new_data)} yeni kayit")
                return {
                    'status': 'updated',
                    'ticker': storage_ticker,
                    'new_records': len(new_data),
                    'total_records': len(combined_data),
                    'last_date': combined_data.index[-1].date().isoformat(),
                    'timestamp': datetime.now().isoformat()
                }

            logger.info(f"INFO {storage_ticker} icin yeni veri yok")
            return {
                'status': 'no_new_data',
                'ticker': storage_ticker,
                'last_date': last_date.date().isoformat(),
                'timestamp': datetime.now().isoformat()
            }

        # İlk kez veri çekme
        new_data = self.fetch_new_data(api_ticker)

        if new_data is not None and not new_data.empty:
            new_data.index = pd.to_datetime(new_data.index)

            # 🔥 CRITICAL FIX: Kolon normalize (doğru metod)
            new_data = self._normalize_columns(new_data)

            new_data = self._add_technical_indicators(new_data)

            file_path = self.get_data_file_path(storage_ticker)
            new_data.to_csv(file_path)

            logger.info(f"OK {storage_ticker} verisi ilk kez indirildi: {len(new_data)} kayit")
            return {
                'status': 'initialized',
                'ticker': storage_ticker,
                'total_records': len(new_data),
                'last_date': new_data.index[-1].date().isoformat(),
                'timestamp': datetime.now().isoformat()
            }

        logger.error(f"HATA {storage_ticker} verisi indirilemedi")
        return {
            'status': 'error',
            'ticker': storage_ticker,
            'timestamp': datetime.now().isoformat()
        }

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Technical indicators ekle"""
        try:
            # Close kolonu bul
            if 'Close' in df.columns:
                close_col = 'Close'
            elif 'close' in df.columns:
                close_col = 'close'
            else:
                logger.error("Close kolonu bulunamadi")
                return df

            # SMA
            df['sma_20'] = df[close_col].rolling(window=20).mean()
            df['sma_50'] = df[close_col].rolling(window=50).mean()

            # EMA
            df['ema_12'] = df[close_col].ewm(span=12).mean()
            df['ema_26'] = df[close_col].ewm(span=26).mean()

            # RSI
            delta = df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            df['bb_middle'] = df[close_col].rolling(window=20).mean()
            bb_std = df[close_col].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # ATR
            high_col = 'High' if 'High' in df.columns else 'high'
            low_col = 'Low' if 'Low' in df.columns else 'low'

            high_low = df[high_col] - df[low_col]
            high_close = np.abs(df[high_col] - df[close_col].shift())
            low_close = np.abs(df[low_col] - df[close_col].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr_14'] = true_range.rolling(14).mean()

            # Lowercase
            df.columns = df.columns.str.lower()

            return df

        except Exception as e:
            logger.error(f"HATA Technical indicators hatasi: {e}")
            return df

    def update_all_stocks(self, stock_list: list) -> dict:
        """Tüm hisseler için güncelle"""
        results = {}
        logger.info(f"GUNCELLEME {len(stock_list)} hisse icin basladi...")

        for ticker in stock_list:
            results[ticker] = self.update_stock(ticker)

        logger.info("OK Tum hisseler guncellendi")
        return results


if __name__ == "__main__":
    updater = DataUpdater()
    result = updater.update_stock('GARAN.IS')
    print(result)