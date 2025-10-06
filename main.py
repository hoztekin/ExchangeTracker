"""
Borsa Trend Analizi - Veri Toplama Modülü
1-2. Hafta: Veri toplama ve keşif
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class StockDataCollector:
    """Yahoo Finance üzerinden hisse senedi verilerini toplayan sınıf"""

    def __init__(self):
        # BIST-30 ve S&P 500'den seçili hisseler
        self.bist30_stocks = [
            'THYAO.IS',  # Türk Hava Yolları
            'AKBNK.IS',  # Akbank
            'GARAN.IS',  # Garanti
            'ISCTR.IS',  # İş Bankası
            'EREGL.IS',  # Ereğli Demir Çelik
            'SAHOL.IS',  # Sabancı Holding
            'KCHOL.IS',  # Koç Holding
            'TUPRS.IS',  # Tüpraş
            'SISE.IS',  # Şişe Cam
            'PETKM.IS'  # Petkim
        ]

        self.sp500_stocks = [
            'AAPL',  # Apple
            'MSFT',  # Microsoft
            'GOOGL',  # Google
            'AMZN',  # Amazon
            'TSLA',  # Tesla
            'NVDA',  # Nvidia
            'META',  # Meta
            'JPM',  # JP Morgan
            'V',  # Visa
            'WMT'  # Walmart
        ]

        self.all_stocks = self.bist30_stocks + self.sp500_stocks

    def fetch_historical_data(self, ticker, period='5y'):
        """
        Belirli bir hisse için tarihsel veri çeker

        Parameters:
        - ticker: Hisse kodu (örn: 'THYAO.IS')
        - period: Veri periyodu (default: 5y)

        Returns:
        - DataFrame: OHLCV verileri
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                print(f"⚠️  {ticker} için veri bulunamadı")
                return None

            # Kolon isimlerini düzenle
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            df['ticker'] = ticker

            print(f"✅ {ticker}: {len(df)} günlük veri çekildi")
            return df

        except Exception as e:
            print(f"❌ {ticker} hatası: {str(e)}")
            return None

    def fetch_all_stocks(self, stock_list=None):
        """Tüm hisseler için veri toplar"""
        if stock_list is None:
            stock_list = self.all_stocks

        all_data = {}

        print(f"\n📊 {len(stock_list)} hisse için veri toplama başlıyor...\n")

        for ticker in stock_list:
            df = self.fetch_historical_data(ticker)
            if df is not None:
                all_data[ticker] = df

        print(f"\n✨ Toplam {len(all_data)} hisse verisi başarıyla toplandı!\n")
        return all_data

    def clean_data(self, df):
        """Veri temizleme ve kalite kontrolü"""
        original_len = len(df)

        # Missing values kontrolü
        missing = df.isnull().sum()
        if missing.any():
            print(f"⚠️  Missing values bulundu:\n{missing[missing > 0]}")
            # Forward fill sonra backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')

        # Outlier detection (IQR yöntemi)
        for col in ['close', 'volume']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"⚠️  {col} için {outliers} outlier tespit edildi")

        # Negatif fiyat kontrolü
        if (df['close'] <= 0).any():
            print("⚠️  Negatif veya sıfır fiyat değerleri bulundu!")
            df = df[df['close'] > 0]

        print(f"✅ Temizleme tamamlandı: {original_len} → {len(df)} satır")
        return df

    def get_data_summary(self, data_dict):
        """Toplanan veriler hakkında özet bilgi"""
        summary = []

        for ticker, df in data_dict.items():
            summary.append({
                'Ticker': ticker,
                'Satır Sayısı': len(df),
                'Başlangıç': df.index.min().strftime('%Y-%m-%d'),
                'Bitiş': df.index.max().strftime('%Y-%m-%d'),
                'Ort. Fiyat': f"${df['close'].mean():.2f}",
                'Ort. Volume': f"{df['volume'].mean():,.0f}",
                'Missing %': f"{(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%"
            })

        return pd.DataFrame(summary)

    def save_data(self, data_dict, output_dir='data'):
        """Verileri CSV formatında kaydet"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        for ticker, df in data_dict.items():
            # Güvenli dosya adı oluştur
            filename = ticker.replace('.', '_').replace('/', '_')
            filepath = f"{output_dir}/{filename}.csv"
            df.to_csv(filepath)
            print(f"💾 {ticker} → {filepath}")

        print(f"\n✅ Tüm veriler '{output_dir}' klasörüne kaydedildi!")


# KULLANIM ÖRNEĞİ
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 BORSA TREND ANALİZİ - VERİ TOPLAMA SİSTEMİ")
    print("=" * 60)

    # Collector oluştur
    collector = StockDataCollector()

    # Önce birkaç hisse ile test edelim
    test_stocks = ['THYAO.IS', 'AAPL', 'MSFT']
    print(f"\n🧪 Test modu: {test_stocks}")

    # Veri topla
    data = collector.fetch_all_stocks(test_stocks)

    # Veri temizleme
    print("\n🧹 Veri temizleme başlıyor...")
    cleaned_data = {}
    for ticker, df in data.items():
        print(f"\n--- {ticker} ---")
        cleaned_data[ticker] = collector.clean_data(df)

    # Özet rapor
    print("\n📋 VERİ ÖZETİ:")
    print("=" * 60)
    summary = collector.get_data_summary(cleaned_data)
    print(summary.to_string(index=False))

    # Verileri kaydet
    print("\n" + "=" * 60)
    collector.save_data(cleaned_data)

    print("\n✨ Veri toplama tamamlandı!")