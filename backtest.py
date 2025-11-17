"""
Backtesting Modülü
Regression modellerinin geçmiş performansını test eder
Kar/zarar, Sharpe Ratio, Max Drawdown hesaplar
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class Backtester:
    """Trading stratejisi backtesting sınıfı"""

    def __init__(self, models_dir='models', data_dir='data/technical'):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.results = {}

    def load_model_components(self, ticker, model_name):
        """Model, scaler ve feature listesini yükle"""
        clean_ticker = ticker.replace('.', '_').replace('^', '').replace('=', '_')

        model_file = self.models_dir / f"{clean_ticker}_{model_name}_model.pkl"
        scaler_file = self.models_dir / f"{clean_ticker}_{model_name}_scaler.pkl"
        features_file = self.models_dir / f"{clean_ticker}_{model_name}_features.pkl"
        metadata_file = self.models_dir / f"{clean_ticker}_{model_name}_metadata.pkl"

        if not all([f.exists() for f in [model_file, scaler_file, features_file]]):
            return None

        return {
            'model': joblib.load(model_file),
            'scaler': joblib.load(scaler_file),
            'features': joblib.load(features_file),
            'metadata': joblib.load(metadata_file)
        }

    def load_historical_data(self, ticker):
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

    def backtest_strategy(self, ticker, model_name,
                          initial_capital=100000,
                          threshold=0.02,
                          commission=0.001,
                          test_period_days=252):
        """
        Trading stratejisi backtest et

        Parameters:
        - ticker: Hisse kodu
        - model_name: Model adı
        - initial_capital: Başlangıç sermayesi ($)
        - threshold: Alım/satım eşiği (0.02 = %2)
        - commission: İşlem komisyonu (0.001 = %0.1)
        - test_period_days: Test dönemi (gün)

        Returns:
        - dict: Backtest sonuçları
        """
        print(f"\n{'=' * 70}")
        print(f"📊 BACKTESTING: {ticker} - {model_name.upper()}")
        print(f"{'=' * 70}\n")

        # Model yükle
        model_data = self.load_model_components(ticker, model_name)
        if model_data is None:
            print(f"❌ Model yüklenemedi")
            return None

        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']

        # Veri yükle
        df = self.load_historical_data(ticker)
        if df is None:
            print(f"❌ Veri yüklenemedi")
            return None

        print(f"📂 Veri yüklendi: {len(df)} gün")

        # Feature hazırla
        X = self.prepare_features(df, features)

        # Target (yarının fiyatı)
        y_true = df['close'].shift(-1)

        # Temizle
        valid_idx = X.notna().all(axis=1) & y_true.notna()
        X = X[valid_idx]
        y_true = y_true[valid_idx]
        df_clean = df[valid_idx]

        # Test periyodu (son N gün)
        X_test = X.tail(test_period_days)
        y_test = y_true.tail(test_period_days)
        df_test = df_clean.tail(test_period_days)

        print(f"📅 Test dönemi: {X_test.index[0].strftime('%Y-%m-%d')} - {X_test.index[-1].strftime('%Y-%m-%d')}")
        print(f"📊 Test günü sayısı: {len(X_test)}")

        # Scale ve tahmin
        X_scaled = scaler.transform(X_test)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0)
        predictions = model.predict(X_scaled)

        # Backtesting simülasyonu
        capital = initial_capital
        position = 0  # 0: pozisyon yok, >0: hisse sayısı
        trades = []
        portfolio_values = []

        for i in range(len(X_test)):
            date = X_test.index[i]
            today_price = df_test.loc[date, 'close']
            predicted_price = predictions[i]

            # Beklenen değişim
            expected_change = (predicted_price - today_price) / today_price

            # Portföy değeri
            if position > 0:
                portfolio_value = capital + (position * today_price)
            else:
                portfolio_value = capital

            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'position': position,
                'price': today_price
            })

            # Trading kararı
            if position == 0:  # Pozisyon yok
                if expected_change > threshold:  # BUY sinyali
                    # Tüm sermaye ile al
                    shares_to_buy = int(capital / (today_price * (1 + commission)))
                    if shares_to_buy > 0:
                        cost = shares_to_buy * today_price * (1 + commission)
                        capital -= cost
                        position = shares_to_buy

                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': today_price,
                            'shares': shares_to_buy,
                            'cost': cost,
                            'expected_change': expected_change,
                            'capital': capital
                        })

            else:  # Pozisyon var
                if expected_change < -threshold:  # SELL sinyali
                    # Tüm pozisyonu sat
                    revenue = position * today_price * (1 - commission)
                    capital += revenue

                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': today_price,
                        'shares': position,
                        'revenue': revenue,
                        'expected_change': expected_change,
                        'capital': capital
                    })

                    position = 0

        # Son durumda pozisyon varsa kapat
        if position > 0:
            final_price = df_test['close'].iloc[-1]
            revenue = position * final_price * (1 - commission)
            capital += revenue

            trades.append({
                'date': X_test.index[-1],
                'action': 'SELL (FINAL)',
                'price': final_price,
                'shares': position,
                'revenue': revenue,
                'expected_change': 0,
                'capital': capital
            })

            position = 0

        # Final portföy değeri
        final_value = capital

        # Performans metrikleri
        total_return = ((final_value - initial_capital) / initial_capital) * 100

        # Buy & Hold karşılaştırması
        buy_hold_shares = int(initial_capital / df_test['close'].iloc[0])
        buy_hold_value = buy_hold_shares * df_test['close'].iloc[-1]
        buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100

        # Portfolio değerleri DataFrame
        df_portfolio = pd.DataFrame(portfolio_values)

        # Daily returns
        df_portfolio['daily_return'] = df_portfolio['portfolio_value'].pct_change()

        # Sharpe Ratio (annualized)
        if df_portfolio['daily_return'].std() > 0:
            sharpe_ratio = (df_portfolio['daily_return'].mean() /
                            df_portfolio['daily_return'].std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Max Drawdown
        df_portfolio['cummax'] = df_portfolio['portfolio_value'].cummax()
        df_portfolio['drawdown'] = (df_portfolio['portfolio_value'] -
                                    df_portfolio['cummax']) / df_portfolio['cummax']
        max_drawdown = df_portfolio['drawdown'].min() * 100

        # Win rate
        winning_trades = sum(1 for t in trades if t['action'].startswith('SELL') and
                             trades.index(t) > 0 and
                             t['price'] > trades[trades.index(t) - 1]['price'])
        total_trades = len([t for t in trades if t['action'].startswith('SELL')])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Sonuçlar
        results = {
            'ticker': ticker,
            'model_name': model_name,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': total_return - buy_hold_return,
            'total_trades': len(trades),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': trades,
            'portfolio_values': df_portfolio,
            'test_period_days': len(X_test)
        }

        # Yazdır
        print(f"\n{'─' * 70}")
        print(f"💰 PERFORMANS SONUÇLARI")
        print(f"{'─' * 70}")
        print(f"   Başlangıç:        ${initial_capital:,.2f}")
        print(f"   Final Değer:      ${final_value:,.2f}")
        print(f"   Toplam Getiri:    {total_return:+.2f}%")
        print(f"   Buy & Hold:       {buy_hold_return:+.2f}%")
        print(f"   Outperformance:   {total_return - buy_hold_return:+.2f}%")

        print(f"\n{'─' * 70}")
        print(f"📊 RİSK METRİKLERİ")
        print(f"{'─' * 70}")
        print(f"   Sharpe Ratio:     {sharpe_ratio:.2f}")
        print(f"   Max Drawdown:     {max_drawdown:.2f}%")

        print(f"\n{'─' * 70}")
        print(f"📈 İŞLEM İSTATİSTİKLERİ")
        print(f"{'─' * 70}")
        print(f"   Toplam İşlem:     {len(trades)}")
        print(f"   Kazanan Oran:     {win_rate:.1f}%")

        # Trade detayları (ilk 5 ve son 5)
        if len(trades) > 0:
            print(f"\n{'─' * 70}")
            print(f"🔍 İLK 5 İŞLEM")
            print(f"{'─' * 70}")
            for trade in trades[:5]:
                action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
                print(f"   {action_emoji} {trade['date'].strftime('%Y-%m-%d')} | "
                      f"{trade['action']:10s} | "
                      f"{trade['shares']:>5} hisse @ ${trade['price']:.2f}")

            if len(trades) > 10:
                print(f"\n   ... ({len(trades) - 10} işlem daha) ...\n")

            print(f"{'─' * 70}")
            print(f"🔍 SON 5 İŞLEM")
            print(f"{'─' * 70}")
            for trade in trades[-5:]:
                action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
                print(f"   {action_emoji} {trade['date'].strftime('%Y-%m-%d')} | "
                      f"{trade['action']:10s} | "
                      f"{trade['shares']:>5} hisse @ ${trade['price']:.2f}")

        print(f"\n{'=' * 70}\n")

        self.results[f"{ticker}_{model_name}"] = results
        return results

    def compare_models(self):
        """Tüm modelleri karşılaştır"""
        if not self.results:
            print("❌ Henüz backtest yapılmadı!")
            return

        print("\n" + "=" * 70)
        print("📊 MODEL KARŞILAŞTIRMASI")
        print("=" * 70 + "\n")

        comparison_data = []
        for key, result in self.results.items():
            comparison_data.append({
                'Model': f"{result['ticker']} - {result['model_name']}",
                'Getiri (%)': result['total_return'],
                'Buy&Hold (%)': result['buy_hold_return'],
                'Fark (%)': result['outperformance'],
                'Sharpe': result['sharpe_ratio'],
                'Max DD (%)': result['max_drawdown'],
                'İşlem': result['total_trades'],
                'Win Rate (%)': result['win_rate']
            })

        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Getiri (%)', ascending=False)

        print(df_comparison.to_string(index=False))

        # En iyi model
        best = df_comparison.iloc[0]
        print(f"\n{'=' * 70}")
        print(f"🏆 EN İYİ MODEL")
        print(f"{'=' * 70}")
        print(f"   Model:       {best['Model']}")
        print(f"   Getiri:      {best['Getiri (%)']:+.2f}%")
        print(f"   Sharpe:      {best['Sharpe']:.2f}")
        print(f"   Max DD:      {best['Max DD (%)']:.2f}%")
        print(f"{'=' * 70}\n")

    def generate_report(self, output_file='outputs/backtest_report.txt'):
        """Detaylı rapor oluştur"""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("📊 BACKTESTING RAPORU\n")
            f.write("=" * 70 + "\n")
            f.write(f"📅 Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"📊 Test Edilen Model Sayısı: {len(self.results)}\n")
            f.write("=" * 70 + "\n\n")

            for key, result in self.results.items():
                f.write("─" * 70 + "\n")
                f.write(f"📌 {result['ticker']} - {result['model_name'].upper()}\n")
                f.write("─" * 70 + "\n\n")

                f.write("💰 PERFORMANS:\n")
                f.write(f"   Başlangıç Sermayesi:  ${result['initial_capital']:,.2f}\n")
                f.write(f"   Final Değer:          ${result['final_value']:,.2f}\n")
                f.write(f"   Toplam Getiri:        {result['total_return']:+.2f}%\n")
                f.write(f"   Buy & Hold Getiri:    {result['buy_hold_return']:+.2f}%\n")
                f.write(f"   Outperformance:       {result['outperformance']:+.2f}%\n\n")

                f.write("📊 RİSK METRİKLERİ:\n")
                f.write(f"   Sharpe Ratio:         {result['sharpe_ratio']:.2f}\n")
                f.write(f"   Max Drawdown:         {result['max_drawdown']:.2f}%\n\n")

                f.write("📈 İŞLEM İSTATİSTİKLERİ:\n")
                f.write(f"   Toplam İşlem:         {result['total_trades']}\n")
                f.write(f"   Kazanan Oran:         {result['win_rate']:.1f}%\n")
                f.write(f"   Test Dönemi:          {result['test_period_days']} gün\n\n")

                # Değerlendirme
                if result['total_return'] > 10 and result['sharpe_ratio'] > 1:
                    grade = "🏆 MÜKEMMEL"
                elif result['total_return'] > 5:
                    grade = "✅ İYİ"
                elif result['total_return'] > 0:
                    grade = "⚠️  ORTA"
                else:
                    grade = "❌ ZAYIF"

                f.write(f"DEĞERLENDİRME: {grade}\n\n")

        print(f"✅ Detaylı rapor kaydedildi: {output_file}\n")


def main():
    """Ana program"""

    print("=" * 70)
    print("📊 BACKTESTING SİSTEMİ")
    print("=" * 70)
    print("\nBu script:")
    print("  • Modelleri geçmiş verilerle test eder")
    print("  • Trading simülasyonu yapar")
    print("  • Kar/zarar hesaplar")
    print("  • Sharpe Ratio, Max Drawdown ölçer")
    print("  • Buy & Hold stratejisi ile karşılaştırır")
    print("=" * 70)

    print("\n⚙️  AYARLAR:")
    print("  • Başlangıç Sermayesi: $100,000")
    print("  • Threshold: ±2% (BUY/SELL)")
    print("  • Komisyon: 0.1%")
    print("  • Test Dönemi: Son 252 gün (~1 yıl)")
    print("=" * 70)

    input("\n▶️  Başlamak için ENTER...")

    backtester = Backtester()

    # Test edilecek modeller her model için özel threshold
    models_to_test = [
        ('AAPL', 'ridge', 0.01),  # ±1% (ABD stabil)
        ('MSFT', 'huber', 0.01),  # ±1% (ABD stabil)
        ('GARAN_IS', 'lassolars', 0.02),  # ±2% (BIST volatil)
        ('THYAO_IS', 'linear', 0.02)  # ±2% (BIST volatil)
    ]

    for ticker, model_name, threshold in models_to_test:
        backtester.backtest_strategy(
            ticker=ticker,
            model_name=model_name,
            initial_capital=100000,
            threshold=threshold,
            commission=0.001,
            test_period_days=252
        )

        input("\n▶️  Sonraki model için ENTER...")

    # Karşılaştırma
    backtester.compare_models()

    # Rapor
    backtester.generate_report()

    print("\n" + "=" * 70)
    print("✅ BACKTESTING TAMAMLANDI!")
    print("=" * 70)
    print("\n📄 Rapor: outputs/backtest_report.txt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem durduruldu.")
    except Exception as e:
        print(f"\n❌ Hata: {str(e)}")
        import traceback

        traceback.print_exc()