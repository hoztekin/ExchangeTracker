"""
Borsa Trend Analizi - Görselleştirme Araçları
Matplotlib ve Seaborn ile profesyonel grafikler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')


class StockVisualizer:
    """Hisse senedi verileri için görselleştirme sınıfı"""

    def __init__(self, figsize=(15, 8), style='seaborn-v0_8-darkgrid'):
        """
        Parameters:
        - figsize: Grafik boyutu
        - style: Matplotlib stil
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.figsize = figsize

    def plot_price_history(self, df, ticker, save_path=None):
        """
        Fiyat geçmişi grafiği (OHLC ve Volume)

        Parameters:
        - df: DataFrame (index: date, columns: open, high, low, close, volume)
        - ticker: Hisse kodu
        - save_path: Kayıt yolu (opsiyonel)
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

        # 1. Fiyat grafiği
        ax1 = fig.add_subplot(gs[0])

        ax1.plot(df.index, df['close'], label='Kapanış', linewidth=2, color='#2E86AB')
        ax1.fill_between(df.index, df['low'], df['high'], alpha=0.2, color='#A23B72', label='High-Low Aralığı')

        ax1.set_title(f'{ticker} - Fiyat Geçmişi', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Tarih', fontsize=12)
        ax1.set_ylabel('Fiyat', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. Volume grafiği
        if 'volume' in df.columns and df['volume'].sum() > 0:
            ax2 = fig.add_subplot(gs[1])

            colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
                      for i in range(len(df))]
            ax2.bar(df.index, df['volume'], color=colors, alpha=0.5)

            ax2.set_xlabel('Tarih', fontsize=12)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Grafik kaydedildi: {save_path}")

        plt.show()

    def plot_candlestick(self, df, ticker, period=60, save_path=None):
        """
        Candlestick (mum) grafiği

        Parameters:
        - df: DataFrame
        - ticker: Hisse kodu
        - period: Gösterilecek gün sayısı
        - save_path: Kayıt yolu
        """
        # Son N günü al
        df = df.tail(period)

        fig, ax = plt.subplots(figsize=self.figsize)

        # Yeşil (yükseliş) ve kırmızı (düşüş) mumlar
        up = df[df['close'] >= df['open']]
        down = df[df['close'] < df['open']]

        # Gövdeler
        ax.bar(up.index, up['close'] - up['open'], bottom=up['open'],
               color='green', alpha=0.8, width=0.6)
        ax.bar(down.index, down['open'] - down['close'], bottom=down['close'],
               color='red', alpha=0.8, width=0.6)

        # Fitiller (high-low çizgileri)
        ax.vlines(up.index, up['low'], up['high'], color='green', linewidth=1)
        ax.vlines(down.index, down['low'], down['high'], color='red', linewidth=1)

        ax.set_title(f'{ticker} - Candlestick Chart (Son {period} Gün)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Tarih', fontsize=12)
        ax.set_ylabel('Fiyat', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_returns_distribution(self, df, ticker, save_path=None):
        """
        Günlük getiri dağılımı

        Parameters:
        - df: DataFrame (daily_return sütunu gerekli)
        - ticker: Hisse kodu
        - save_path: Kayıt yolu
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # 1. Histogram
        axes[0].hist(df['daily_return'].dropna(), bins=50, alpha=0.7,
                     color='steelblue', edgecolor='black')
        axes[0].axvline(df['daily_return'].mean(), color='red',
                        linestyle='--', linewidth=2, label=f'Ortalama: {df["daily_return"].mean():.3f}%')
        axes[0].set_title(f'{ticker} - Günlük Getiri Dağılımı', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Günlük Getiri (%)', fontsize=12)
        axes[0].set_ylabel('Frekans', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Box plot
        axes[1].boxplot(df['daily_return'].dropna(), vert=True)
        axes[1].set_title(f'{ticker} - Box Plot', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Günlük Getiri (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_correlation_heatmap(self, correlation_matrix, title='Korelasyon Matrisi', save_path=None):
        """
        Korelasyon ısı haritası

        Parameters:
        - correlation_matrix: Korelasyon matrisi
        - title: Başlık
        - save_path: Kayıt yolu
        """
        fig, ax = plt.subplots(figsize=(14, 12))

        # Heatmap
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1, ax=ax)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_volatility_comparison(self, data_dict, save_path=None):
        """
        Hisseler arası volatilite karşılaştırması

        Parameters:
        - data_dict: {ticker: DataFrame} dictionary
        - save_path: Kayıt yolu
        """
        volatilities = {}

        for ticker, df in data_dict.items():
            if 'daily_return' in df.columns:
                volatilities[ticker] = df['daily_return'].std()

        # Sıralama
        sorted_vols = dict(sorted(volatilities.items(), key=lambda x: x[1], reverse=True))

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = ['red' if '_IS' in t else 'blue' for t in sorted_vols.keys()]
        bars = ax.barh(list(sorted_vols.keys()), list(sorted_vols.values()), color=colors, alpha=0.7)

        ax.set_xlabel('Volatilite (Std Sapma)', fontsize=12)
        ax.set_title('Hisseler Arası Volatilite Karşılaştırması', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')

        # Renk açıklaması
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='BIST'),
                           Patch(facecolor='blue', alpha=0.7, label='S&P 500')]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_cumulative_returns(self, data_dict, tickers, save_path=None):
        """
        Kümülatif getiri karşılaştırması

        Parameters:
        - data_dict: {ticker: DataFrame} dictionary
        - tickers: Karşılaştırılacak ticker listesi
        - save_path: Kayıt yolu
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for ticker in tickers:
            if ticker in data_dict:
                df = data_dict[ticker]

                # Normalize edilmiş fiyat (başlangıç = 100)
                normalized = (df['close'] / df['close'].iloc[0]) * 100
                ax.plot(df.index, normalized, label=ticker, linewidth=2)

        ax.set_title('Kümülatif Getiri Karşılaştırması (Başlangıç = 100)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Tarih', fontsize=12)
        ax.set_ylabel('Normalize Fiyat', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_volume_price_relationship(self, df, ticker, save_path=None):
        """
        Volume-Price ilişkisi scatter plot

        Parameters:
        - df: DataFrame
        - ticker: Hisse kodu
        - save_path: Kayıt yolu
        """
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            print(f"⚠️  {ticker} için volume verisi yok!")
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        # Fiyat değişimine göre renklendirme
        colors = df['daily_return'].apply(lambda x: 'green' if x > 0 else 'red')

        scatter = ax.scatter(df['volume'], df['daily_return'],
                             c=colors, alpha=0.5, s=30)

        ax.set_xlabel('Volume', fontsize=12)
        ax.set_ylabel('Günlük Getiri (%)', fontsize=12)
        ax.set_title(f'{ticker} - Volume vs Fiyat Değişimi',
                     fontsize=16, fontweight='bold', pad=20)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_seasonal_patterns(self, df, ticker, save_path=None):
        """
        Mevsimsel paternler (aylık ve günlük)

        Parameters:
        - df: DataFrame
        - ticker: Hisse kodu
        - save_path: Kayıt yolu
        """
        df = df.copy()
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # 1. Aylık ortalama getiri
        monthly_returns = df.groupby('month')['daily_return'].mean()
        months = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz',
                  'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara']

        colors_monthly = ['green' if x > 0 else 'red' for x in monthly_returns.values]
        axes[0].bar(monthly_returns.index, monthly_returns.values, color=colors_monthly, alpha=0.7)
        axes[0].set_xticks(monthly_returns.index)
        axes[0].set_xticklabels([months[i - 1] for i in monthly_returns.index], rotation=45)
        axes[0].set_title('Aylık Ortalama Getiri', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Ortalama Getiri (%)', fontsize=12)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0].grid(True, alpha=0.3, axis='y')

        # 2. Haftanın gününe göre getiri
        day_returns = df[df['day_of_week'] < 5].groupby('day_of_week')['daily_return'].mean()
        days = ['Pzt', 'Sal', 'Çar', 'Per', 'Cum']

        colors_daily = ['green' if x > 0 else 'red' for x in day_returns.values]

        axes[1].bar(day_returns.index, day_returns.values, color=colors_daily, alpha=0.7)
        axes[1].set_xticks(day_returns.index)
        axes[1].set_xticklabels([days[i] for i in day_returns.index])
        axes[1].set_title('Haftanın Gününe Göre Ortalama Getiri', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Ortalama Getiri (%)', fontsize=12)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'{ticker} - Mevsimsel Paternler', fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_risk_return_scatter(self, data_dict, save_path=None):
        """
        Risk-Return scatter plot (Volatilite vs Getiri)

        Parameters:
        - data_dict: {ticker: DataFrame} dictionary
        - save_path: Kayıt yolu
        """
        returns = []
        risks = []
        tickers = []
        colors = []

        for ticker, df in data_dict.items():
            if 'daily_return' in df.columns:
                avg_return = df['daily_return'].mean()
                risk = df['daily_return'].std()

                returns.append(avg_return)
                risks.append(risk)
                tickers.append(ticker)
                colors.append('red' if '_IS' in ticker else 'blue')

        fig, ax = plt.subplots(figsize=self.figsize)

        scatter = ax.scatter(risks, returns, c=colors, alpha=0.6, s=100)

        # Etiketler
        for i, ticker in enumerate(tickers):
            ax.annotate(ticker, (risks[i], returns[i]), fontsize=8, alpha=0.7)

        ax.set_xlabel('Risk (Volatilite)', fontsize=12)
        ax.set_ylabel('Ortalama Günlük Getiri (%)', fontsize=12)
        ax.set_title('Risk-Return Analizi', fontsize=16, fontweight='bold', pad=20)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(risks), color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        # Renk açıklaması
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.6, label='BIST'),
                           Patch(facecolor='blue', alpha=0.6, label='S&P 500')]
        ax.legend(handles=legend_elements, loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


if __name__ == "__main__":
    print("📊 Visualization module hazır!")
    print("Kullanım: from src.utils.visualization import StockVisualizer")