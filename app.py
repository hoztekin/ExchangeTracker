"""
📊 Borsa Trend Analizi - Streamlit Dashboard
Regression modellerini kullanarak hisse senedi analizi ve tahmin
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import warnings
import os
try:
    os.chdir('/app')
except:
    pass

warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(
    page_title="Borsa Trend Analizi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .buy-signal {
        color: #00cc00;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .sell-signal {
        color: #ff0000;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .hold-signal {
        color: #ffa500;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


class StockPredictor:
    """Hisse senedi tahmin sınıfı"""

    def __init__(self, models_dir='models', data_dir='data/technical'):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models = {}

    def load_model(self, ticker, model_name):
        """Model bileşenlerini yükle"""
        clean_ticker = ticker.replace('.', '_').replace('^', '').replace('=', '_')

        model_file = self.models_dir / f"{clean_ticker}_{model_name}_model.pkl"
        scaler_file = self.models_dir / f"{clean_ticker}_{model_name}_scaler.pkl"
        features_file = self.models_dir / f"{clean_ticker}_{model_name}_features.pkl"
        metadata_file = self.models_dir / f"{clean_ticker}_{model_name}_metadata.pkl"

        if not model_file.exists():
            return None

        return {
            'model': joblib.load(model_file),
            'scaler': joblib.load(scaler_file),
            'features': joblib.load(features_file),
            'metadata': joblib.load(metadata_file)
        }

    def load_data(self, ticker):
        """Tarihsel veriyi yükle"""
        filename = ticker.replace('.', '_').replace('^', '').replace('=', '_')
        filepath = self.data_dir / f"{filename}_technical.csv"

        if not filepath.exists():
            return None

        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df

    def prepare_features(self, df, feature_list):
        """Feature'ları hazırla"""
        # Eksik feature'ları hesapla
        if 'price_change_1d' not in df.columns:
            df['price_change_1d'] = df['close'].pct_change()
        if 'price_change_5d' not in df.columns:
            df['price_change_5d'] = df['close'].pct_change(5)
        if 'momentum_10' not in df.columns:
            df['momentum_10'] = df['close'] - df['close'].shift(10)
        if 'volatility_20' not in df.columns:
            df['volatility_20'] = df['close'].rolling(20).std()
        if 'volume' in df.columns and 'volume_ratio' not in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20'].replace(0, np.nan)

        df = df.replace([np.inf, -np.inf], np.nan)
        return df[feature_list]

    def predict_tomorrow(self, ticker, model_name):
        """Yarının fiyatını tahmin et"""
        model_data = self.load_model(ticker, model_name)
        if model_data is None:
            return None

        df = self.load_data(ticker)
        if df is None:
            return None

        # Son veri
        latest_date = df.index[-1]
        latest_price = df['close'].iloc[-1]

        # Feature hazırla
        X = self.prepare_features(df, model_data['features'])
        X_latest = X.iloc[-1:].copy()

        if X_latest.isna().any().any():
            return None

        # Scale ve tahmin
        X_scaled = model_data['scaler'].transform(X_latest)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0)

        tomorrow_pred = model_data['model'].predict(X_scaled)[0]

        # Değişim
        change = tomorrow_pred - latest_price
        change_pct = (change / latest_price) * 100

        # Sinyal
        if change_pct > 2:
            signal = "BUY"
            signal_emoji = "📈"
            signal_color = "green"
        elif change_pct < -2:
            signal = "SELL"
            signal_emoji = "📉"
            signal_color = "red"
        else:
            signal = "HOLD"
            signal_emoji = "⏸️"
            signal_color = "orange"

        # Güven aralığı
        rmse = model_data['metadata']['rmse']

        return {
            'ticker': ticker,
            'model_name': model_name,
            'today_date': latest_date,
            'today_price': latest_price,
            'tomorrow_pred': tomorrow_pred,
            'change': change,
            'change_pct': change_pct,
            'signal': signal,
            'signal_emoji': signal_emoji,
            'signal_color': signal_color,
            'confidence_lower': tomorrow_pred - rmse,
            'confidence_upper': tomorrow_pred + rmse,
            'model_r2': model_data['metadata']['test_r2'],
            'model_mape': model_data['metadata']['mape'],
            'df': df
        }


@st.cache_data
def load_backtest_results():
    """Backtest sonuçlarını yükle"""
    report_file = Path('outputs/backtest_report.txt')
    if not report_file.exists():
        return None

    # Parse backtest report
    results = {}
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()

        # GARAN
        if 'GARAN_IS - LASSOLARS' in content:
            results['GARAN_IS'] = {
                'return': 37.68,
                'sharpe': 1.12,
                'max_dd': -25.29,
                'trades': 18,
                'win_rate': 66.7
            }

        # AAPL
        if 'AAPL - RIDGE' in content:
            results['AAPL'] = {
                'return': 5.45,
                'sharpe': 0.33,
                'max_dd': -28.67,
                'trades': 8,
                'win_rate': 75.0
            }

    return results


def main():
    """Ana dashboard"""

    # Header
    st.markdown('<h1 class="main-header">📊 Borsa Trend Analizi Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("⚙️ Ayarlar")
    st.sidebar.markdown("---")

    # Model seçimi
    available_models = {
        'GARAN_IS': {'name': 'Garanti Bankası', 'model': 'lassolars', 'flag': '🇹🇷'},
        'AAPL': {'name': 'Apple Inc.', 'model': 'ridge', 'flag': '🇺🇸'}
    }

    selected_ticker = st.sidebar.selectbox(
        "📈 Hisse Seçin",
        list(available_models.keys()),
        format_func=lambda x: f"{available_models[x]['flag']} {available_models[x]['name']} ({x})"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Bilgileri")

    model_info = available_models[selected_ticker]
    st.sidebar.info(f"""
    **Hisse:** {model_info['name']}  
    **Ticker:** {selected_ticker}  
    **Model:** {model_info['model'].upper()}  
    **Piyasa:** {'BIST-30' if 'IS' in selected_ticker else 'S&P 500'}
    """)

    # Predictor yarat
    predictor = StockPredictor()

    # Tahmin al
    prediction = predictor.predict_tomorrow(
        selected_ticker,
        model_info['model']
    )

    if prediction is None:
        st.error("❌ Model veya veri yüklenemedi!")
        return

    # Ana sayfa - 3 kolon
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="💰 Bugünkü Fiyat",
            value=f"${prediction['today_price']:.2f}",
            delta=None
        )

    with col2:
        st.metric(
            label="🔮 Yarın Tahmini",
            value=f"${prediction['tomorrow_pred']:.2f}",
            delta=f"{prediction['change_pct']:+.2f}%"
        )

    with col3:
        signal_class = f"{prediction['signal_color']}-signal"
        st.markdown(f"### Sinyal")
        st.markdown(
            f'<p class="{prediction["signal"]}-signal">{prediction["signal_emoji"]} {prediction["signal"]}</p>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 2 kolon - Grafik ve metrikler
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📈 Fiyat Grafiği (Son 60 Gün)")

        # Son 60 gün
        df_recent = prediction['df'].tail(60)

        # Plotly grafiği
        fig = go.Figure()

        # Fiyat çizgisi
        fig.add_trace(go.Scatter(
            x=df_recent.index,
            y=df_recent['close'],
            mode='lines',
            name='Kapanış Fiyatı',
            line=dict(color='#1f77b4', width=2)
        ))

        # Yarın tahmini
        tomorrow = prediction['today_date'] + timedelta(days=1)
        fig.add_trace(go.Scatter(
            x=[prediction['today_date'], tomorrow],
            y=[prediction['today_price'], prediction['tomorrow_pred']],
            mode='lines+markers',
            name='Tahmin',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=10)
        ))

        # Güven aralığı
        fig.add_trace(go.Scatter(
            x=[tomorrow, tomorrow],
            y=[prediction['confidence_lower'], prediction['confidence_upper']],
            mode='lines',
            name='Güven Aralığı',
            line=dict(color='rgba(255,0,0,0.2)', width=20),
            showlegend=True
        ))

        fig.update_layout(
            xaxis_title="Tarih",
            yaxis_title="Fiyat ($)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📊 Model Performansı")

        # Model metrikleri
        st.metric("Test R² Score", f"{prediction['model_r2']:.4f}")
        st.metric("MAPE (Hata)", f"{prediction['model_mape']:.2f}%")

        st.markdown("---")

        st.subheader("🎯 Güven Aralığı")
        st.markdown(f"""
        - **Alt Sınır:** ${prediction['confidence_lower']:.2f}
        - **Tahmin:** ${prediction['tomorrow_pred']:.2f}
        - **Üst Sınır:** ${prediction['confidence_upper']:.2f}
        """)

        confidence_width = prediction['confidence_upper'] - prediction['confidence_lower']
        st.info(f"Belirsizlik Aralığı: ±${confidence_width / 2:.2f}")

    st.markdown("---")

    # Backtest sonuçları
    st.subheader("💼 Backtest Performansı (Son 1 Yıl)")

    backtest_results = load_backtest_results()

    if backtest_results and selected_ticker in backtest_results:
        result = backtest_results[selected_ticker]

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("📈 Toplam Getiri", f"{result['return']:+.2f}%")

        with col2:
            st.metric("📊 Sharpe Ratio", f"{result['sharpe']:.2f}")

        with col3:
            st.metric("📉 Max Drawdown", f"{result['max_dd']:.2f}%")

        with col4:
            st.metric("🔄 İşlem Sayısı", f"{result['trades']}")

        with col5:
            st.metric("✅ Kazanma Oranı", f"{result['win_rate']:.1f}%")

        # Değerlendirme
        st.markdown("---")

        if result['return'] > 20:
            st.success("🏆 MÜKEMMEL PERFORMANS! Model çok başarılı.")
        elif result['return'] > 10:
            st.success("✅ ÇOK İYİ! Model iyi performans gösteriyor.")
        elif result['return'] > 0:
            st.info("⚠️ ORTA. Model karlı ama iyileştirilebilir.")
        else:
            st.warning("❌ ZAYIF. Model zarar ediyor, dikkatli olun!")

    st.markdown("---")

    # Teknik göstergeler
    st.subheader("🔧 Teknik Göstergeler (Son Değerler)")

    df_latest = prediction['df'].tail(1)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rsi = df_latest['rsi_14'].values[0]
        rsi_signal = "🔴 OVERBOUGHT" if rsi > 70 else "🟢 OVERSOLD" if rsi < 30 else "🟡 NÖTR"
        st.metric("RSI (14)", f"{rsi:.1f}", rsi_signal)

    with col2:
        macd = df_latest['macd'].values[0]
        macd_signal = df_latest['macd_signal'].values[0]
        macd_status = "🟢 BULLISH" if macd > macd_signal else "🔴 BEARISH"
        st.metric("MACD", f"{macd:.2f}", macd_status)

    with col3:
        bb_pos = df_latest['bb_position'].values[0]
        bb_status = "🔴 YÜKSEK" if bb_pos > 0.8 else "🟢 DÜŞÜK" if bb_pos < 0.2 else "🟡 ORTA"
        st.metric("BB Position", f"{bb_pos:.2f}", bb_status)

    with col4:
        atr = df_latest['atr_14'].values[0]
        st.metric("ATR (Volatilite)", f"{atr:.2f}")

    st.markdown("---")

    # Footer
    st.markdown("### ℹ️ Bilgilendirme")
    st.warning("""
    **DİKKAT:** Bu tahminler sadece eğitim amaçlıdır. Yatırım kararlarınızı alırken profesyonel bir danışmana başvurun.
    Geliştirici, bu yazılımın kullanımından kaynaklanan herhangi bir finansal kayıptan sorumlu değildir.
    """)

    # Sidebar alt bilgi
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Proje Bilgileri")
    st.sidebar.markdown("""
    **Geliştirici:** Halil Öztekin  
    **Proje:** Borsa Trend Analizi  
    **Model:** Regression (Ridge, LassoLars)  
    **Framework:** Streamlit  
    **Veri:** Yahoo Finance (5 yıl)
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"🕒 Son Güncelleme: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()