"""
Borsa Trend Analizi - Keşifsel Veri Analizi (EDA)
3-4. Hafta: Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ExploratoryDataAnalysis:
    """Hisse senedi verileri için kapsamlı EDA sınıfı"""

    def __init__(self, data_dir='data'):
        """
        Parameters:
        - data_dir: CSV dosyalarının bulunduğu klasör
        """
        self.data_dir = Path(data_dir)
        self.data = {}
        self.stats = {}

        # Stil ayarları
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def load_data(self, tickers=None):
        """CSV dosyalarını yükle"""
        print("📂 Veriler yükleniyor...\n")

        csv_files = list(self.data_dir.glob('*.csv'))

        if not csv_files:
            print(f"❌ '{self.data_dir}' klasöründe CSV dosyası bulunamadı!")
            return

        for csv_file in csv_files:
            ticker = csv_file.stem  # Dosya adı (uzantısız)

            # Eğer belirli tickerlar istenmişse filtrele
            if tickers and ticker not in tickers:
                continue

            try:
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'], utc=True)
                # Timezone'ı temizle (timezone-aware datetime sorunlarını önler)
                df['date'] = df['date'].dt.tz_localize(None)
                df = df.set_index('date')

                self.data[ticker] = df
                print(f"✅ {ticker:15s} → {len(df)} satır")

            except Exception as e:
                print(f"❌ {ticker}: {str(e)}")

        print(f"\n📊 Toplam {len(self.data)} sembol yüklendi\n")

    def calculate_basic_stats(self):
        """Temel istatistiksel metrikleri hesapla"""
        print("📊 Temel istatistikler hesaplanıyor...\n")

        stats_list = []

        for ticker, df in self.data.items():
            # Günlük getiri hesapla
            df['daily_return'] = df['close'].pct_change() * 100

            # Volatilite (20 günlük)
            df['volatility_20'] = df['daily_return'].rolling(20).std()

            stats = {
                'Ticker': ticker,
                'Başlangıç': df.index.min().strftime('%Y-%m-%d'),
                'Bitiş': df.index.max().strftime('%Y-%m-%d'),
                'Gün Sayısı': len(df),
                'Başlangıç Fiyat': df['close'].iloc[0],
                'Son Fiyat': df['close'].iloc[-1],
                'Değişim %': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
                'Min Fiyat': df['close'].min(),
                'Max Fiyat': df['close'].max(),
                'Ort. Günlük Getiri %': df['daily_return'].mean(),
                'Std Günlük Getiri %': df['daily_return'].std(),
                'Ort. Volatilite': df['volatility_20'].mean(),
                'Ort. Volume': df['volume'].mean() if 'volume' in df.columns else 0
            }

            stats_list.append(stats)
            self.stats[ticker] = stats

        stats_df = pd.DataFrame(stats_list)
        return stats_df

    def analyze_price_movements(self, ticker):
        """Fiyat hareketlerini detaylı analiz et"""
        if ticker not in self.data:
            print(f"❌ {ticker} verisi bulunamadı!")
            return

        df = self.data[ticker]

        print(f"\n{'='*70}")
        print(f"📈 {ticker} - FİYAT HAREKETLERİ ANALİZİ")
        print(f"{'='*70}\n")

        # 1. Temel metrikler
        print("📊 TEMEL METRİKLER:")
        print(f"   • Dönem: {df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}")
        print(f"   • Gün Sayısı: {len(df)}")
        print(f"   • Başlangıç Fiyat: {df['close'].iloc[0]:.2f}")
        print(f"   • Son Fiyat: {df['close'].iloc[-1]:.2f}")
        print(f"   • Toplam Değişim: {((df['close'].iloc[-1]/df['close'].iloc[0])-1)*100:.2f}%")

        # 2. En yüksek/düşük
        print(f"\n📍 FİYAT ARALIKLARI:")
        max_date = df['close'].idxmax()
        min_date = df['close'].idxmin()
        print(f"   • En Yüksek: {df['close'].max():.2f} ({max_date.strftime('%Y-%m-%d')})")
        print(f"   • En Düşük: {df['close'].min():.2f} ({min_date.strftime('%Y-%m-%d')})")
        print(f"   • Fark: {((df['close'].max()/df['close'].min())-1)*100:.2f}%")

        # 3. Günlük getiri analizi
        print(f"\n💰 GÜNLÜK GETİRİ ANALİZİ:")
        print(f"   • Ortalama: {df['daily_return'].mean():.3f}%")
        print(f"   • Medyan: {df['daily_return'].median():.3f}%")
        print(f"   • Std Sapma: {df['daily_return'].std():.3f}%")
        print(f"   • En Büyük Kazanç: {df['daily_return'].max():.2f}%")
        print(f"   • En Büyük Kayıp: {df['daily_return'].min():.2f}%")

        # 4. Volatilite
        print(f"\n📊 VOLATİLİTE:")
        print(f"   • Ortalama 20-günlük: {df['volatility_20'].mean():.3f}%")
        print(f"   • Maksimum: {df['volatility_20'].max():.3f}%")
        print(f"   • Minimum: {df['volatility_20'].min():.3f}%")

        # 5. Volume analizi (varsa)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            print(f"\n📦 VOLUME ANALİZİ:")
            print(f"   • Ortalama: {df['volume'].mean():,.0f}")
            print(f"   • Maksimum: {df['volume'].max():,.0f}")
            print(f"   • Minimum: {df['volume'].min():,.0f}")

        print(f"\n{'='*70}\n")

    def calculate_correlation_matrix(self, market='all'):
        """
        Hisseler arası korelasyon matrisi hesapla

        Parameters:
        - market: 'bist', 'sp500', 'all'
        """
        print(f"🔗 Korelasyon matrisi hesaplanıyor ({market})...\n")

        # Piyasa filtreleme
        if market == 'bist':
            tickers = [t for t in self.data.keys() if '_IS' in t]
        elif market == 'sp500':
            tickers = [t for t in self.data.keys()
                      if '_IS' not in t and not any(x in t for x in ['TRY', 'VIX', 'GC', 'CL', 'XU100', 'GSPC'])]
        else:
            tickers = list(self.data.keys())

        # Kapanış fiyatlarını birleştir
        close_prices = pd.DataFrame()
        for ticker in tickers:
            if ticker in self.data:
                close_prices[ticker] = self.data[ticker]['close']

        # Korelasyon hesapla
        correlation = close_prices.corr()

        return correlation

    def analyze_volume_price_relationship(self, ticker):
        """Volume-Price ilişkisini analiz et"""
        if ticker not in self.data:
            print(f"❌ {ticker} verisi bulunamadı!")
            return

        df = self.data[ticker]

        if 'volume' not in df.columns or df['volume'].sum() == 0:
            print(f"⚠️  {ticker} için volume verisi yok!")
            return

        print(f"\n{'='*70}")
        print(f"📊 {ticker} - VOLUME-PRICE İLİŞKİSİ")
        print(f"{'='*70}\n")

        # Fiyat değişimi kategorileri
        df['price_change'] = df['close'].pct_change() * 100
        df['volume_change'] = df['volume'].pct_change() * 100

        # Korelasyon
        corr = df[['price_change', 'volume']].corr().iloc[0, 1]
        print(f"📈 Fiyat Değişimi - Volume Korelasyonu: {corr:.3f}")

        # Volume kategorileri
        df['volume_category'] = pd.qcut(df['volume'], q=4, labels=['Düşük', 'Orta', 'Yüksek', 'Çok Yüksek'])

        print(f"\n📦 VOLUME KATEGORİLERİNE GÖRE ORTALAMA FİYAT DEĞİŞİMİ:")
        volume_analysis = df.groupby('volume_category')['price_change'].agg(['mean', 'std', 'count'])
        print(volume_analysis)

        print(f"\n{'='*70}\n")

    def detect_seasonal_patterns(self, ticker):
        """Mevsimsel/dönemsel paternleri tespit et"""
        if ticker not in self.data:
            print(f"❌ {ticker} verisi bulunamadı!")
            return

        df = self.data[ticker].copy()

        print(f"\n{'='*70}")
        print(f"📅 {ticker} - MEVSİMSEL PATERNLER")
        print(f"{'='*70}\n")

        # Tarih bileşenlerini ekle (index zaten DatetimeIndex)
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        df['quarter'] = df.index.quarter

        # Aylık analiz
        print("📊 AYLIK ORTALAMA GETİRİ:")
        monthly_returns = df.groupby('month')['daily_return'].mean()
        for month, ret in monthly_returns.items():
            month_name = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz',
                         'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara'][month-1]
            print(f"   {month_name}: {ret:>7.3f}%")

        # Haftanın günü analizi
        print(f"\n📅 HAFTANIN GÜNÜNE GÖRE ORTALAMA GETİRİ:")
        day_returns = df.groupby('day_of_week')['daily_return'].mean()
        days = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma']
        for day, ret in day_returns.items():
            if day < 5:  # Sadece hafta içi
                print(f"   {days[day]:12s}: {ret:>7.3f}%")

        # Çeyreklik analiz
        print(f"\n📈 ÇEYREKLİK ORTALAMA GETİRİ:")
        quarterly_returns = df.groupby('quarter')['daily_return'].mean()
        for q, ret in quarterly_returns.items():
            print(f"   Q{q}: {ret:>7.3f}%")

        print(f"\n{'='*70}\n")

    def compare_markets(self):
        """BIST ve S&P 500 piyasalarını karşılaştır"""
        print(f"\n{'='*70}")
        print(f"🌍 PİYASA KARŞILAŞTIRMASI: BIST vs S&P 500")
        print(f"{'='*70}\n")

        # BIST hisseleri
        bist_tickers = [t for t in self.data.keys() if '_IS' in t]
        # S&P 500 hisseleri
        sp500_tickers = [t for t in self.data.keys()
                        if '_IS' not in t and not any(x in t for x in ['TRY', 'VIX', 'GC', 'CL', 'XU100', 'GSPC'])]

        if not bist_tickers or not sp500_tickers:
            print("⚠️  Karşılaştırma için yeterli veri yok!")
            return

        # BIST ortalamaları
        bist_returns = []
        bist_vols = []
        for ticker in bist_tickers:
            if ticker in self.data:
                bist_returns.append(self.data[ticker]['daily_return'].mean())
                bist_vols.append(self.data[ticker]['daily_return'].std())

        # S&P 500 ortalamaları
        sp500_returns = []
        sp500_vols = []
        for ticker in sp500_tickers:
            if ticker in self.data:
                sp500_returns.append(self.data[ticker]['daily_return'].mean())
                sp500_vols.append(self.data[ticker]['daily_return'].std())

        print(f"📊 BIST-30 ({len(bist_tickers)} hisse):")
        print(f"   • Ort. Günlük Getiri: {np.mean(bist_returns):.3f}%")
        print(f"   • Ort. Volatilite: {np.mean(bist_vols):.3f}%")

        print(f"\n📊 S&P 500 ({len(sp500_tickers)} hisse):")
        print(f"   • Ort. Günlük Getiri: {np.mean(sp500_returns):.3f}%")
        print(f"   • Ort. Volatilite: {np.mean(sp500_vols):.3f}%")

        print(f"\n💡 FARKLAR:")
        print(f"   • Getiri Farkı: {np.mean(bist_returns) - np.mean(sp500_returns):.3f}%")
        print(f"   • Volatilite Farkı: {np.mean(bist_vols) - np.mean(sp500_vols):.3f}%")

        print(f"\n{'='*70}\n")

    def generate_summary_report(self):
        """Kapsamlı özet rapor oluştur"""
        print("\n" + "="*70)
        print("📋 KAPSAMLI EDA ÖZET RAPORU")
        print("="*70 + "\n")

        # 1. Genel bilgiler
        print("📊 GENEL BİLGİLER:")
        print(f"   • Toplam Sembol: {len(self.data)}")
        bist = len([t for t in self.data.keys() if '_IS' in t])
        sp500 = len([t for t in self.data.keys()
                    if '_IS' not in t and not any(x in t for x in ['TRY', 'VIX', 'GC', 'CL', 'XU100', 'GSPC'])])
        print(f"   • BIST Hisseleri: {bist}")
        print(f"   • S&P 500 Hisseleri: {sp500}")
        print(f"   • Makro/Endeks: {len(self.data) - bist - sp500}")

        # 2. En iyi/kötü performans
        print(f"\n🏆 PERFORMANS LİDERLERİ:")
        returns = {ticker: ((self.data[ticker]['close'].iloc[-1] /
                            self.data[ticker]['close'].iloc[0]) - 1) * 100
                  for ticker in self.data.keys()}

        sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)

        print(f"\n   En İyi 5:")
        for ticker, ret in sorted_returns[:5]:
            print(f"   • {ticker:12s}: {ret:>7.2f}%")

        print(f"\n   En Kötü 5:")
        for ticker, ret in sorted_returns[-5:]:
            print(f"   • {ticker:12s}: {ret:>7.2f}%")

        # 3. Volatilite analizi
        print(f"\n📊 VOLATİLİTE ANALİZİ:")
        vols = {ticker: self.data[ticker]['daily_return'].std()
               for ticker in self.data.keys()}
        sorted_vols = sorted(vols.items(), key=lambda x: x[1], reverse=True)

        print(f"\n   En Volatil 5:")
        for ticker, vol in sorted_vols[:5]:
            print(f"   • {ticker:12s}: {vol:>7.3f}%")

        print(f"\n   En Stabil 5:")
        for ticker, vol in sorted_vols[-5:]:
            print(f"   • {ticker:12s}: {vol:>7.3f}%")

        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Örnek kullanım
    eda = ExploratoryDataAnalysis(data_dir='data')

    # Verileri yükle
    eda.load_data()

    # Temel istatistikler
    stats = eda.calculate_basic_stats()
    print(stats.to_string(index=False))

    # Örnek analizler
    eda.analyze_price_movements('THYAO_IS')
    eda.compare_markets()
    eda.generate_summary_report()