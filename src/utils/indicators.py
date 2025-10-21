"""
Borsa Trend Analizi - Teknik Gösterge Hesaplama Modülü
5-7. Hafta: Technical Indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class TechnicalIndicators:
    """Teknik göstergeleri hesaplayan utility sınıfı"""

    # ==================== MOVING AVERAGES ====================

    @staticmethod
    def calculate_sma(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Simple Moving Average (SMA) hesapla

        Parameters:
        - data: Kapanış fiyatı (pd.Series)
        - period: Periyot (gün)

        Returns:
        - SMA değerleri
        """
        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Exponential Moving Average (EMA) hesapla
        Son değerlere daha fazla ağırlık verir

        Parameters:
        - data: Kapanış fiyatı
        - period: Periyot

        Returns:
        - EMA değerleri
        """
        return data.ewm(span=period, adjust=False).mean()

    # ==================== MOMENTUM ====================

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI) hesapla

        RSI = 100 - (100 / (1 + RS))
        RS = Ort. Kazanç / Ort. Kayıp

        Yorumlama:
        - RSI > 70: Overbought (Satış sinyali)
        - RSI < 30: Oversold (Alış sinyali)

        Parameters:
        - data: Kapanış fiyatı
        - period: Periyot (genelde 14)

        Returns:
        - RSI değerleri (0-100)
        """
        # Fiyat değişimlerini hesapla
        delta = data.diff()

        # Kazanç ve kayıpları ayır
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Ortalama kazanç/kayıp
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # RS hesapla (sıfıra bölme hatasını önle)
        rs = avg_gain / avg_loss.replace(0, np.nan)

        # RSI hesapla
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series,
                             close: pd.Series, period: int = 14,
                             smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator hesapla

        K% = ((Close - Low) / (High - Low)) × 100
        D% = K%'nin SMA'sı

        Yorumlama:
        - K > 80: Overbought
        - K < 20: Oversold
        - K > D: Alış sinyali (bullish)
        - K < D: Satış sinyali (bearish)

        Parameters:
        - high: En yüksek fiyat
        - low: En düşük fiyat
        - close: Kapanış fiyatı
        - period: Periyot
        - smooth: Smoothing periyotu

        Returns:
        - (K%, D%) tuple'ı
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=smooth).mean()

        return k_percent, d_percent

    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series,
                             close: pd.Series, period: int = 14) -> pd.Series:
        """
        Williams %R hesapla

        %R = -100 × ((High - Close) / (High - Low))

        Yorumlama:
        - %R > -20: Overbought
        - %R < -80: Oversold
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        r_percent = -100 * (highest_high - close) / (highest_high - lowest_low)

        return r_percent

    # ==================== TREND ====================

    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12,
                       slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence) hesapla

        MACD = EMA(12) - EMA(26)
        Signal = EMA(MACD, 9)
        Histogram = MACD - Signal

        Yorumlama:
        - MACD > Signal: Alış sinyali
        - MACD < Signal: Satış sinyali
        - Histogram > 0: Momentum artıyor

        Parameters:
        - data: Kapanış fiyatı
        - fast: Hızlı EMA periyodu
        - slow: Yavaş EMA periyodu
        - signal: Signal line periyodu

        Returns:
        - (MACD, Signal, Histogram) tuple'ı
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal

        return macd, macd_signal, macd_hist

    # ==================== VOLATİLİTE ====================

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20,
                                  std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands hesapla

        Middle = SMA(20)
        Upper = Middle + (Std × 2)
        Lower = Middle - (Std × 2)

        Yorumlama:
        - Price > Upper: Overbought
        - Price < Lower: Oversold
        - Bands genişliyor: Volatilite artıyor
        - Bands darlaşıyor: Breakout gelmek üzere

        Parameters:
        - data: Kapanış fiyatı
        - period: Ortalama periyodu
        - std_dev: Standart sapma çarpanı

        Returns:
        - (Upper, Middle, Lower) tuple'ı
        """
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series,
                      close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR) hesapla
        Volatilitenin bir ölçüsü

        TR = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
        ATR = EMA(TR, period)

        Yorumlama:
        - Yüksek ATR: Yüksek volatilite
        - Düşük ATR: Düşük volatilite
        """
        # True Range hesapla
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR hesapla (EMA)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    # ==================== VOLUME ====================

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume (OBV) hesapla

        Cumulative volume indicator
        OBV = Prev_OBV + Volume (eğer Close > Prev_Close)
        OBV = Prev_OBV - Volume (eğer Close < Prev_Close)
        OBV = Prev_OBV (eğer Close = Prev_Close)

        Yorumlama:
        - OBV yükseliş trendi: Alış baskısı
        - OBV düşüş trendi: Satış baskısı
        """
        # Fiyat değişimlerini hesapla
        price_diff = close.diff()

        # Volume işaretini belirle
        obv = volume.copy()
        obv = obv.where(price_diff > 0, -obv)
        obv = obv.where(price_diff != 0, 0)

        # Kümülatif topla
        obv = obv.cumsum()

        return obv

    @staticmethod
    def calculate_mfi(high: pd.Series, low: pd.Series,
                      close: pd.Series, volume: pd.Series,
                      period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI) hesapla

        Typical Price = (High + Low + Close) / 3
        Money Flow = Typical Price × Volume

        MFI = 100 - (100 / (1 + Money Flow Ratio))

        Yorumlama:
        - MFI > 80: Overbought
        - MFI < 20: Oversold
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        # Pozitif/Negatif flow
        positive_flow = money_flow.where(
            typical_price > typical_price.shift(1), 0
        )
        negative_flow = money_flow.where(
            typical_price < typical_price.shift(1), 0
        )

        # Toplama
        positive_flow_sum = positive_flow.rolling(window=period).sum()
        negative_flow_sum = negative_flow.rolling(window=period).sum()

        # MFI hesapla
        mfi_ratio = positive_flow_sum / negative_flow_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfi_ratio))

        return mfi

    # ==================== DESTEK/DİRENÇ ====================

    @staticmethod
    def calculate_pivot_points(high: pd.Series, low: pd.Series,
                               close: pd.Series, period: int = 5) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Pivot Points hesapla (Klasik Method)

        Pivot = (High + Low + Close) / 3
        Resistance1 = (2 × Pivot) - Low
        Support1 = (2 × Pivot) - High

        Parameters:
        - period: Kaç gün öncesini referans al

        Returns:
        - (Pivot, Resistance, Support) tuple'ı
        """
        # Period günlük high/low'u al
        high_period = high.rolling(window=period).max()
        low_period = low.rolling(window=period).min()

        # Pivot
        pivot = (high_period + low_period + close) / 3

        # Resistance ve Support
        resistance = (2 * pivot) - low_period
        support = (2 * pivot) - high_period

        return pivot, resistance, support

    @staticmethod
    def calculate_local_extremes(high: pd.Series, low: pd.Series,
                                 window: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Yerel yüksek ve düşükler hesapla

        Local High: Çevresindeki değerlerden daha yüksek
        Local Low: Çevresindeki değerlerden daha düşük

        Parameters:
        - window: Karşılaştırma penceresi (2×window+1)

        Returns:
        - (Local_High, Local_Low) boolean series'i
        """
        # Yerel yüksek
        local_high = (high == high.rolling(
            window=2 * window + 1, center=True
        ).max())

        # Yerel düşük
        local_low = (low == low.rolling(
            window=2 * window + 1, center=True
        ).min())

        return local_high, local_low

    # ==================== SİNYAL ÜRETİMİ ====================

    @staticmethod
    def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm göstergelere dayalı BUY/SELL/HOLD sinyalleri üret

        Parameters:
        - df: Tüm teknik göstergelerin hesaplandığı DataFrame

        Returns:
        - DataFrame with signal columns
        """
        df = df.copy()

        # Signal skorunu başlat (0-1 arası)
        buy_score = pd.Series(0.0, index=df.index)
        sell_score = pd.Series(0.0, index=df.index)

        # ===== RSI Sinyalleri (0.2 ağırlık) =====
        buy_score += (df['rsi_14'] < 30).astype(float) * 0.2
        sell_score += (df['rsi_14'] > 70).astype(float) * 0.2

        # ===== MACD Sinyalleri (0.2 ağırlık) =====
        macd_buy = (df['macd'] > df['macd_signal']) & \
                   (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        macd_sell = (df['macd'] < df['macd_signal']) & \
                    (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        buy_score += macd_buy.astype(float) * 0.2
        sell_score += macd_sell.astype(float) * 0.2

        # ===== Bollinger Bands Sinyalleri (0.15 ağırlık) =====
        bb_buy = df['close'] < df['bb_lower']
        bb_sell = df['close'] > df['bb_upper']
        buy_score += bb_buy.astype(float) * 0.15
        sell_score += bb_sell.astype(float) * 0.15

        # ===== SMA Crossover Sinyalleri (0.15 ağırlık) =====
        sma_buy = (df['sma_20'] > df['sma_50']) & \
                  (df['sma_20'].shift(1) <= df['sma_50'].shift(1))
        sma_sell = (df['sma_20'] < df['sma_50']) & \
                   (df['sma_20'].shift(1) >= df['sma_50'].shift(1))
        buy_score += sma_buy.astype(float) * 0.15
        sell_score += sma_sell.astype(float) * 0.15

        # ===== Stochastic Sinyalleri (0.15 ağırlık) =====
        stoch_buy = (df['stochastic_k'] < 20) & (df['stochastic_k'] > df['stochastic_d'])
        stoch_sell = (df['stochastic_k'] > 80) & (df['stochastic_k'] < df['stochastic_d'])
        buy_score += stoch_buy.astype(float) * 0.15
        sell_score += stoch_sell.astype(float) * 0.15

        # ===== Final Sinyaller =====
        df['buy_signal_score'] = buy_score
        df['sell_signal_score'] = sell_score

        # Signal türü belirle
        df['signal'] = 'HOLD'
        df.loc[buy_score > 0.5, 'signal'] = 'BUY'
        df.loc[sell_score > 0.5, 'signal'] = 'SELL'
        df.loc[(buy_score > 0.5) & (sell_score > 0.5), 'signal'] = 'HOLD'  # Conflict

        # Signal gücü (0-1)
        df['signal_strength'] = np.maximum(buy_score, sell_score)

        return df


if __name__ == "__main__":
    print("📊 Technical Indicators Module Hazır!")
    print("\nKullanılan Göstergeler:")
    print("  • Moving Averages: SMA, EMA")
    print("  • Momentum: RSI, Stochastic, Williams %R")
    print("  • Trend: MACD")
    print("  • Volatilite: Bollinger Bands, ATR")
    print("  • Volume: OBV, MFI")
    print("  • Support/Resistance: Pivot Points, Local Extremes")
    print("  • Sinyal Üretimi: Multi-indicator")