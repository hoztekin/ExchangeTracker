"""
Borsa Trend Analizi - Veri Toplama Modülü
1-2. Hafta: Veri toplama ve keşif
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')


class StockDataCollector:
    """Yahoo Finance üzerinden hisse senedi verilerini toplayan sınıf"""

    def __init__(self):
        # BIST-30 hisseleri
        self.bist30_stocks = [
            'THYAO.IS',  # Türk Hava Yolları
            'AKBNK.IS',  # Akbank
            'GARAN.IS',  # Garanti Bankası
            'ISCTR.IS',  # İş Bankası (C)
            'EREGL.IS',  # Ereğli Demir Çelik
            'SAHOL.IS',  # Sabancı Holding
            'KCHOL.IS',  # Koç Holding
            'TUPRS.IS',  # Tüpraş
            'SISE.IS',   # Şişe Cam
            'PETKM.IS'   # Petkim
        ]

        # S&P 500 hisseleri
        self.sp500_stocks = [
            'AAPL',   # Apple
            'MSFT',   # Microsoft
            'GOOGL',  # Google
            'AMZN',   # Amazon
            'TSLA',   # Tesla
            'NVDA',   # Nvidia
            'META',   # Meta (Facebook)
            'JPM',    # JP Morgan
            'V',      # Visa
            'WMT'     # Walmart
        ]

        # Makroekonomik veriler
        self.macro_data = [
            'TRY=X',   # USD/TRY Kuru (BIST için kritik!)
            '^VIX',    # Volatilite Endeksi
            'GC=F',    # Altın
            'CL=F'     # Ham Petrol
        ]

        # Endeksler
        self.indices = [
            '^XU100',  # BIST 100
            '^GSPC'    # S&P 500
        ]

        # Tüm semboller
        self.all_stocks = (
            self.bist30_stocks +
            self.sp500_stocks +
            self.macro_data +
            self.indices
        )

    def fetch_historical_data(self, ticker, period='5y'):
        """
        Belirli bir hisse/sembol için tarihsel veri çeker

        Parameters:
        - ticker: Hisse kodu (örn: 'THYAO.IS', 'TRY=X')
        - period: Veri periyodu (default: 5y)

        Returns:
        - DataFrame: OHLCV verileri
        """
        try:
            print(f"📥 {ticker} verisi çekiliyor...", end=" ")

            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                print(f"❌ Veri bulunamadı")
                return None

            # Kolon isimlerini düzenle
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            df['ticker'] = ticker

            # Tarih indexini sütun yap
            df.reset_index(inplace=True)
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]

            print(f"✅ {len(df)} gün")
            return df

        except Exception as e:
            print(f"❌ Hata: {str(e)}")
            return None

    def fetch_all_stocks(self, stock_list=None, retry_failed=True):
        """
        Tüm hisseler için veri toplar

        Parameters:
        - stock_list: Özel hisse listesi (None ise tümü)
        - retry_failed: Başarısız olanları tekrar dene
        """
        if stock_list is None:
            stock_list = self.all_stocks

        all_data = {}
        failed = []

        print("\n" + "="*70)
        print(f"📊 BORSA VERİ TOPLAMA BAŞLIYOR - {len(stock_list)} SEMBOL")
        print("="*70 + "\n")

        # İlk deneme
        for ticker in stock_list:
            df = self.fetch_historical_data(ticker)
            if df is not None:
                all_data[ticker] = df
            else:
                failed.append(ticker)

        # Başarısız olanları tekrar dene
        if retry_failed and failed:
            print(f"\n🔄 {len(failed)} başarısız sembol tekrar deneniyor...")
            for ticker in failed[:]:
                df = self.fetch_historical_data(ticker)
                if df is not None:
                    all_data[ticker] = df
                    failed.remove(ticker)

        # Özet
        print("\n" + "="*70)
        print(f"✅ BAŞARILI: {len(all_data)}/{len(stock_list)}")
        if failed:
            print(f"❌ BAŞARISIZ: {len(failed)} → {', '.join(failed)}")
        print("="*70 + "\n")

        return all_data

    def clean_data(self, df):
        """
        Veri temizleme işlemleri

        - Missing values kontrolü
        - Outlier detection
        - Veri tutarlılık kontrolleri
        """
        if df is None or df.empty:
            return None

        ticker = df['ticker'].iloc[0]
        original_len = len(df)

        print(f"\n🧹 {ticker} temizleniyor...")

        # 1. Missing values kontrolü
        missing = df.isnull().sum()
        if missing.any():
            print(f"   ⚠️  Missing values: {missing[missing > 0].to_dict()}")
            # Forward fill sonra backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')

        # 2. Negatif fiyat kontrolü
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    print(f"   ⚠️  {col}: {negative_count} negatif değer düzeltildi")
                    df[col] = df[col].abs()

        # 3. High-Low tutarlılık kontrolü
        if 'high' in df.columns and 'low' in df.columns:
            invalid = df['high'] < df['low']
            if invalid.any():
                print(f"   ⚠️  {invalid.sum()} satırda high < low anomalisi düzeltildi")
                df.loc[invalid, 'high'], df.loc[invalid, 'low'] = (
                    df.loc[invalid, 'low'], df.loc[invalid, 'high']
                )

        # 4. Outlier detection (IQR method)
        if 'close' in df.columns:
            Q1 = df['close'].quantile(0.25)
            Q3 = df['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR

            outliers = ((df['close'] < lower) | (df['close'] > upper)).sum()
            if outliers > 0:
                print(f"   📊 {outliers} outlier tespit edildi (korundu)")

        # 5. Duplicate kontrolü
        duplicates = df.duplicated(subset=['date']).sum()
        if duplicates > 0:
            print(f"   ⚠️  {duplicates} duplicate kayıt silindi")
            df = df.drop_duplicates(subset=['date'], keep='first')

        cleaned_len = len(df)
        print(f"   ✅ {original_len} → {cleaned_len} satır")

        return df

    def get_data_summary(self, data_dict):
        """Toplanan veriler için özet rapor oluştur"""
        summary_data = []

        for ticker, df in data_dict.items():
            if df is None or df.empty:
                continue

            summary_data.append({
                'Ticker': ticker,
                'Satır Sayısı': len(df),
                'Başlangıç': df['date'].min().strftime('%Y-%m-%d'),
                'Bitiş': df['date'].max().strftime('%Y-%m-%d'),
                'Missing %': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
                'Ort. Volume': f"{df['volume'].mean():,.0f}" if 'volume' in df.columns else 'N/A'
            })

        return pd.DataFrame(summary_data)

    def save_data(self, data_dict, output_dir='data'):
        """Verileri CSV formatında kaydet"""

        # Klasör oluştur
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n💾 Veriler '{output_dir}/' klasörüne kaydediliyor...\n")

        for ticker, df in data_dict.items():
            if df is None or df.empty:
                continue

            # Dosya adını düzenle (özel karakterleri temizle)
            filename = ticker.replace('^', '').replace('=', '_').replace('.', '_')
            filepath = f"{output_dir}/{filename}.csv"

            df.to_csv(filepath, index=False)
            print(f"✅ {ticker:12s} → {filepath}")

        print(f"\n📁 Tüm veriler '{output_dir}/' klasörüne kaydedildi!")


def main():
    """Ana program akışı"""

    print("="*70)
    print("🚀 BORSA TREND ANALİZİ - VERİ TOPLAMA SİSTEMİ")
    print("="*70)
    print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Hedef: 5 yıllık tarihsel veri (2020-2025)")
    print("="*70)

    # Collector oluştur
    collector = StockDataCollector()

    print(f"\n📋 TOPLANACAK VERİLER:")
    print(f"   • BIST-30: {len(collector.bist30_stocks)} hisse")
    print(f"   • S&P 500: {len(collector.sp500_stocks)} hisse")
    print(f"   • Makro: {len(collector.macro_data)} sembol")
    print(f"   • Endeks: {len(collector.indices)} sembol")
    print(f"   ─────────────────────")
    print(f"   📊 TOPLAM: {len(collector.all_stocks)} sembol\n")

    input("🔵 Başlamak için ENTER'a basın... ")

    # 1. VERİ TOPLAMA
    print("\n" + "="*70)
    print("1️⃣  VERİ TOPLAMA AŞAMASI")
    print("="*70)

    all_data = collector.fetch_all_stocks()

    if not all_data:
        print("\n❌ Hiç veri toplanamadı! Program sonlandırılıyor.")
        return

    # 2. VERİ TEMİZLEME
    print("\n" + "="*70)
    print("2️⃣  VERİ TEMİZLEME AŞAMASI")
    print("="*70)

    cleaned_data = {}
    for ticker, df in all_data.items():
        cleaned = collector.clean_data(df)
        if cleaned is not None:
            cleaned_data[ticker] = cleaned

    # 3. ÖZET RAPOR
    print("\n" + "="*70)
    print("3️⃣  VERİ ÖZETİ")
    print("="*70)

    summary = collector.get_data_summary(cleaned_data)
    print("\n" + summary.to_string(index=False))

    # 4. VERİLERİ KAYDET
    print("\n" + "="*70)
    print("4️⃣  VERİ KAYDETME")
    print("="*70)

    collector.save_data(cleaned_data)

    # Final özet
    print("\n" + "="*70)
    print("✨ VERİ TOPLAMA SÜRECİ TAMAMLANDI!")
    print("="*70)
    print(f"✅ Başarılı: {len(cleaned_data)} sembol")
    print(f"📁 Konum: ./data/ klasörü")
    print(f"📊 Toplam: {sum(len(df) for df in cleaned_data.values()):,} satır veri")
    print("="*70)
    print("\n🎯 Sonraki adım: EDA (Exploratory Data Analysis) - 3-4. Hafta")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()