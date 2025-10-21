"""
Borsa Trend Analizi - Teknik Analiz Modülü
5-7. Hafta: Technical Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from src.utils.indicators import TechnicalIndicators

warnings.filterwarnings('ignore')


class TechnicalAnalysis:
    """Hisse senedi verileri için kapsamlı teknik analiz sınıfı"""

    def __init__(self, data_dir='data'):
        """
        Parameters:
        - data_dir: CSV dosyalarının bulunduğu klasör
        """
        self.data_dir = Path(data_dir)
        self.data = {}
        self.technical_data = {}
        self.indicators = TechnicalIndicators()

    def load_data(self, tickers=None):
        """CSV dosyalarını yükle"""
        print("📂 Veriler yükleniyor...\n")

        csv_files = list(self.data_dir.glob('*.csv'))

        if not csv_files:
            print(f"❌ '{self.data_dir}' klasöründe CSV dosyası bulunamadı!")
            return

        for csv_file in csv_files:
            ticker = csv_file.stem

            if tickers and ticker not in tickers:
                continue

            try:
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'], utc=True)
                df['date'] = df['date'].dt.tz_localize(None)
                df = df.set_index('date')

                self.data[ticker] = df
                print(f"✅ {ticker:15s} → {len(df)} satır")

            except Exception as e:
                print(f"❌ {ticker}: {str(e)}")

        print(f"\n📊 Toplam {len(self.data)} sembol yüklendi\n")

    def calculate_all_indicators(self, ticker):
        """
        Bir hisse için tüm teknik göstergeleri hesapla

        Parameters:
        - ticker: Hisse kodu

        Returns:
        - DataFrame with all indicators
        """
        if ticker not in self.data:
            print(f"❌ {ticker} verisi bulunamadı!")
            return None

        df = self.data[ticker].copy()

        print(f"🔧 {ticker} göstergeleri hesaplanıyor...")

        # ===== MOVING AVERAGES =====
        df['sma_20'] = self.indicators.calculate_sma(df['close'], period=20)
        df['sma_50'] = self.indicators.calculate_sma(df['close'], period=50)
        df['sma_200'] = self.indicators.calculate_sma(df['close'], period=200)
        df['ema_12'] = self.indicators.calculate_ema(df['close'], period=12)
        df['ema_26'] = self.indicators.calculate_ema(df['close'], period=26)

        # ===== MOMENTUM =====
        df['rsi_14'] = self.indicators.calculate_rsi(df['close'], period=14)
        df['stochastic_k'], df['stochastic_d'] = self.indicators.calculate_stochastic(
            df['high'], df['low'], df['close'], period=14, smooth=3
        )
        df['williams_r'] = self.indicators.calculate_williams_r(
            df['high'], df['low'], df['close'], period=14
        )

        # ===== TREND =====
        df['macd'], df['macd_signal'], df['macd_hist'] = self.indicators.calculate_macd(
            df['close'], fast=12, slow=26, signal=9
        )

        # ===== VOLATİLİTE =====
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.indicators.calculate_bollinger_bands(
            df['close'], period=20, std_dev=2
        )
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width'].replace(0, np.nan)
        df['atr_14'] = self.indicators.calculate_atr(df['high'], df['low'], df['close'], period=14)

        # ===== VOLUME =====
        df['obv'] = self.indicators.calculate_obv(df['close'], df['volume'])
        df['obv_signal'] = self.indicators.calculate_ema(df['obv'], period=9)
        df['mfi_14'] = self.indicators.calculate_mfi(
            df['high'], df['low'], df['close'], df['volume'], period=14
        )

        # ===== DESTEK/DİRENÇ =====
        df['pivot'], df['resistance'], df['support'] = self.indicators.calculate_pivot_points(
            df['high'], df['low'], df['close'], period=5
        )
        df['local_high'], df['local_low'] = self.indicators.calculate_local_extremes(
            df['high'], df['low'], window=5
        )

        # ===== SİNYAL ÜRETİMİ =====
        df = self.indicators.generate_signals(df)

        self.technical_data[ticker] = df
        print(f"✅ {ticker} göstergeleri hesaplandı\n")

        return df

    def calculate_all_tickers(self):
        """Tüm yüklenmiş hisseler için göstergeleri hesapla"""
        print("=" * 70)
        print("🔧 TEKNİK GÖSTERGELERİ HESAPLANIYOR")
        print("=" * 70 + "\n")

        for ticker in self.data.keys():
            self.calculate_all_indicators(ticker)

    def analyze_indicators(self, ticker):
        """Bir hisse için teknik göstergeler analizi"""
        if ticker not in self.technical_data:
            print(f"❌ {ticker} için teknik göstergeler hesaplanmadı!")
            return

        df = self.technical_data[ticker]

        print("\n" + "=" * 70)
        print(f"📊 {ticker} - TEKNİK GÖSTERGELER ANALİZİ")
        print("=" * 70 + "\n")

        # 1. Son fiyat ve hareketler
        print("📈 ŞIMDIKI FIYAT VE TREND:")
        print(f"   • Kapanış: {df['close'].iloc[-1]:.2f}")
        print(f"   • SMA(20): {df['sma_20'].iloc[-1]:.2f}")
        print(f"   • SMA(50): {df['sma_50'].iloc[-1]:.2f}")
        print(f"   • SMA(200): {df['sma_200'].iloc[-1]:.2f}")

        trend = "⬆️ YÜKSELIŞ" if df['close'].iloc[-1] > df['sma_200'].iloc[-1] else "⬇️ DÜŞÜŞ"
        print(f"   • Genel Trend: {trend}")

        # 2. RSI Analizi
        print(f"\n📊 RSI ANALİZİ:")
        rsi = df['rsi_14'].iloc[-1]
        print(f"   • RSI(14): {rsi:.2f}")

        if rsi > 70:
            print(f"   • Durum: 🔴 OVERBOUGHT (Satış sinyali)")
        elif rsi < 30:
            print(f"   • Durum: 🟢 OVERSOLD (Alış sinyali)")
        else:
            print(f"   • Durum: 🟡 NÖTR")

        # 3. MACD Analizi
        print(f"\n📊 MACD ANALİZİ:")
        macd = df['macd'].iloc[-1]
        signal = df['macd_signal'].iloc[-1]
        hist = df['macd_hist'].iloc[-1]

        print(f"   • MACD: {macd:.4f}")
        print(f"   • Signal: {signal:.4f}")
        print(f"   • Histogram: {hist:.4f}")

        if macd > signal:
            print(f"   • Sinyal: 🟢 BUY (MACD > Signal)")
        else:
            print(f"   • Sinyal: 🔴 SELL (MACD < Signal)")

        # 4. Bollinger Bands Analizi
        print(f"\n📊 BOLLINGER BANDS ANALİZİ:")
        close = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_pos = df['bb_position'].iloc[-1]

        print(f"   • Upper: {bb_upper:.2f}")
        print(f"   • Middle: {df['bb_middle'].iloc[-1]:.2f}")
        print(f"   • Lower: {bb_lower:.2f}")
        print(f"   • Close: {close:.2f}")
        print(f"   • Position: {bb_pos:.2%}")

        if close > bb_upper:
            print(f"   • Durum: 🔴 Price > Upper (Overbought)")
        elif close < bb_lower:
            print(f"   • Durum: 🟢 Price < Lower (Oversold)")
        else:
            print(f"   • Durum: 🟡 Inside Bands")

        # 5. Stochastic Analizi
        print(f"\n📊 STOCHASTIC ANALİZİ:")
        k = df['stochastic_k'].iloc[-1]
        d = df['stochastic_d'].iloc[-1]

        print(f"   • K%: {k:.2f}")
        print(f"   • D%: {d:.2f}")

        if k > 80:
            print(f"   • Durum: 🔴 OVERBOUGHT")
        elif k < 20:
            print(f"   • Durum: 🟢 OVERSOLD")
        else:
            print(f"   • Durum: 🟡 NÖTR")

        # 6. ATR (Volatilite)
        print(f"\n📊 VOLATİLİTE (ATR):")
        atr = df['atr_14'].iloc[-1]
        atr_percent = (atr / close) * 100

        print(f"   • ATR(14): {atr:.2f}")
        print(f"   • ATR %: {atr_percent:.2f}%")

        if atr_percent > 3:
            print(f"   • Durum: 🔴 YÜKSEK VOLATİLİTE")
        elif atr_percent < 1:
            print(f"   • Durum: 🟢 DÜŞÜK VOLATİLİTE")
        else:
            print(f"   • Durum: 🟡 NORMAL VOLATİLİTE")

        # 7. Destek/Direnç
        print(f"\n📊 DESTEK VE DİRENÇ:")
        pivot = df['pivot'].iloc[-1]
        support = df['support'].iloc[-1]
        resistance = df['resistance'].iloc[-1]

        print(f"   • Direnç: {resistance:.2f}")
        print(f"   • Pivot: {pivot:.2f}")
        print(f"   • Destek: {support:.2f}")

        if close > resistance:
            print(f"   • Fiyat Konumu: ⬆️ Dirençin Üzerinde")
        elif close < support:
            print(f"   • Fiyat Konumu: ⬇️ Desteğin Altında")
        else:
            print(f"   • Fiyat Konumu: 🟡 Destek-Direnç Arasında")

        # 8. OBV (Volume)
        print(f"\n📊 ON-BALANCE VOLUME (OBV):")
        obv = df['obv'].iloc[-1]
        obv_signal = df['obv_signal'].iloc[-1]

        print(f"   • OBV: {obv:,.0f}")
        print(f"   • OBV Signal: {obv_signal:,.0f}")

        if obv > obv_signal:
            print(f"   • Durum: 🟢 Alış Baskısı")
        else:
            print(f"   • Durum: 🔴 Satış Baskısı")

        # 9. MFI (Money Flow)
        print(f"\n📊 MONEY FLOW INDEX (MFI):")
        mfi = df['mfi_14'].iloc[-1]

        print(f"   • MFI(14): {mfi:.2f}")

        if mfi > 80:
            print(f"   • Durum: 🔴 OVERBOUGHT")
        elif mfi < 20:
            print(f"   • Durum: 🟢 OVERSOLD")
        else:
            print(f"   • Durum: 🟡 NÖTR")

        # 10. Sinyal Özeti
        print(f"\n" + "=" * 70)
        print("🎯 SINYAL ÖZETİ")
        print("=" * 70)

        signal = df['signal'].iloc[-1]
        strength = df['signal_strength'].iloc[-1]

        print(f"\n📊 Son Sinyal: {signal}")
        print(f"📊 Sinyal Gücü: {strength:.2%}")

        if signal == "BUY":
            print(f"\n✅ ÖNERİ: ALMALISINIZ")
        elif signal == "SELL":
            print(f"\n❌ ÖNERİ: SATMALISINIZ")
        else:
            print(f"\n⏸️  ÖNERİ: BEKLEMELİSİNİZ")

        print("\n" + "=" * 70 + "\n")

    def get_signal_summary(self):
        """Tüm hisseler için sinyal özeti"""
        print("\n" + "=" * 70)
        print("📊 GENEL SİNYAL ÖZETİ")
        print("=" * 70 + "\n")

        buy_signals = []
        sell_signals = []
        hold_signals = []

        for ticker, df in self.technical_data.items():
            signal = df['signal'].iloc[-1]
            strength = df['signal_strength'].iloc[-1]

            if signal == "BUY":
                buy_signals.append((ticker, strength))
            elif signal == "SELL":
                sell_signals.append((ticker, strength))
            else:
                hold_signals.append((ticker, strength))

        # Sıralama
        buy_signals.sort(key=lambda x: x[1], reverse=True)
        sell_signals.sort(key=lambda x: x[1], reverse=True)
        hold_signals.sort(key=lambda x: x[1], reverse=True)

        # Yazdırma
        if buy_signals:
            print("🟢 ALMALISINIZ (BUY):")
            for ticker, strength in buy_signals[:5]:
                print(f"   • {ticker:12s} → Gücü: {strength:.2%}")

        if sell_signals:
            print(f"\n🔴 SATMALISINIZ (SELL):")
            for ticker, strength in sell_signals[:5]:
                print(f"   • {ticker:12s} → Gücü: {strength:.2%}")

        if hold_signals:
            print(f"\n🟡 BEKLEMELİSİNİZ (HOLD):")
            for ticker, strength in hold_signals[:5]:
                print(f"   • {ticker:12s} → Gücü: {strength:.2%}")

        print(f"\n" + "=" * 70)
        print(f"📊 ÖZET: {len(buy_signals)} BUY | {len(sell_signals)} SELL | {len(hold_signals)} HOLD")
        print("=" * 70 + "\n")

    def save_technical_data(self, output_dir='data/technical'):
        """Teknik göstergeleri CSV'ye kaydet"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n💾 Teknik veriler '{output_dir}/' klasörüne kaydediliyor...\n")

        for ticker, df in self.technical_data.items():
            filename = ticker.replace('^', '').replace('=', '_').replace('.', '_')
            filepath = f"{output_dir}/{filename}_technical.csv"

            df.to_csv(filepath)
            print(f"✅ {ticker:12s} → {filepath}")

        print(f"\n📁 Tüm veriler kaydedildi!")


if __name__ == "__main__":
    ta = TechnicalAnalysis(data_dir='data')
    ta.load_data()
    ta.calculate_all_tickers()
    ta.get_signal_summary()