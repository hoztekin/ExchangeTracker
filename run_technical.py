"""
Borsa Trend Analizi - Teknik Analiz Çalıştırıcı
5-7. Hafta: Technical Analysis Runner - TAMAMLANDI
"""

import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

from src.analysis.technical import TechnicalAnalysis
from src.utils.visualization import StockVisualizer


def save_summary_report(ta, output_file='outputs/technical_summary_report.txt'):
    """Teknik analiz özet raporunu dosyaya kaydet"""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("📊 BORSA TREND ANALİZİ - TEKNİK ANALİZ RAPORU\n")
        f.write("=" * 70 + "\n")
        f.write(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        buy_count = sum(1 for df in ta.technical_data.values() if df['signal'].iloc[-1] == 'BUY')
        sell_count = sum(1 for df in ta.technical_data.values() if df['signal'].iloc[-1] == 'SELL')
        hold_count = sum(1 for df in ta.technical_data.values() if df['signal'].iloc[-1] == 'HOLD')

        f.write("📊 SİNYAL İSTATİSTİKLERİ\n")
        f.write("=" * 70 + "\n")
        f.write(f"🟢 BUY Sinyalleri: {buy_count}\n")
        f.write(f"🔴 SELL Sinyalleri: {sell_count}\n")
        f.write(f"🟡 HOLD Sinyalleri: {hold_count}\n")
        f.write(f"Toplam: {buy_count + sell_count + hold_count}\n\n")

        f.write("📊 EN GÜÇLÜ BUY SİNYALLERİ\n")
        f.write("=" * 70 + "\n")
        buy_signals = [(ticker, df['signal_strength'].iloc[-1])
                       for ticker, df in ta.technical_data.items()
                       if df['signal'].iloc[-1] == 'BUY']
        buy_signals.sort(key=lambda x: x[1], reverse=True)
        for ticker, strength in buy_signals[:10]:
            f.write(f"   • {ticker:12s} → {strength:.1%}\n")

        f.write("\n📊 EN GÜÇLÜ SELL SİNYALLERİ\n")
        f.write("=" * 70 + "\n")
        sell_signals = [(ticker, df['signal_strength'].iloc[-1])
                        for ticker, df in ta.technical_data.items()
                        if df['signal'].iloc[-1] == 'SELL']
        sell_signals.sort(key=lambda x: x[1], reverse=True)
        for ticker, strength in sell_signals[:10]:
            f.write(f"   • {ticker:12s} → {strength:.1%}\n")

        avg_rsi = np.mean([df['rsi_14'].iloc[-1]
                          for df in ta.technical_data.values()
                          if not pd.isna(df['rsi_14'].iloc[-1])])
        f.write(f"\n\n📊 ORTALAMA METRİKLER\n")
        f.write("=" * 70 + "\n")
        f.write(f"Ort. RSI(14): {avg_rsi:.2f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("✨ RAPOR TAMAMLANDI\n")
        f.write("=" * 70 + "\n")

    print(f"✅ Rapor kaydedildi: {output_file}\n")


def main():
    """Ana teknik analiz işlem akışı"""

    print("=" * 70)
    print("📊 BORSA TREND ANALİZİ - TEKNİK ANALİZ")
    print("=" * 70)
    print("5-7. Hafta: Technical Indicators & Signals")
    print("=" * 70 + "\n")

    ta = TechnicalAnalysis(data_dir='data')
    viz = StockVisualizer()

    # ===== 1. VERİ YÜKLEME =====
    print("=" * 70)
    print("1️⃣  VERİ YÜKLEME")
    print("=" * 70 + "\n")

    try:
        ta.load_data()
        if not ta.data:
            print("❌ Veri yüklenemedi! Önce main.py'yi çalıştırın.")
            return
        print("✅ Veri yükleme başarılı\n")
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}\n")
        return

    input("▶️  Devam etmek için ENTER'a basın...")

    # ===== 2. TEKNİK GÖSTERGELERİ HESAPLA =====
    print("\n" + "=" * 70)
    print("2️⃣  TEKNİK GÖSTERGELERİ HESAPLAMA")
    print("=" * 70 + "\n")

    try:
        ta.calculate_all_tickers()
        print("✅ Göstergeler hesaplandı\n")
    except Exception as e:
        print(f"❌ Gösterge hesaplama hatası: {e}\n")
        return

    input("▶️  Devam etmek için ENTER'a basın...")

    # ===== 3. SİNYAL ÖZETİ =====
    print("\n" + "=" * 70)
    print("3️⃣  GENEL SİNYAL ÖZETİ")
    print("=" * 70 + "\n")

    try:
        ta.get_signal_summary()
    except Exception as e:
        print(f"❌ Sinyal özeti hatası: {e}\n")

    input("▶️  Devam etmek için ENTER'a basın...")

    # ===== 4. DETAYLI ANALİZ =====
    print("\n" + "=" * 70)
    print("4️⃣  DETAYLI ANALİZ - ÖRNEK HİSSELER")
    print("=" * 70 + "\n")

    example_tickers = []
    for ticker in ta.technical_data.keys():
        if '_IS' in ticker and len(example_tickers) < 1:
            example_tickers.append(ticker)
    for ticker in ta.technical_data.keys():
        if '_IS' not in ticker and len(example_tickers) < 2:
            example_tickers.append(ticker)
    for ticker in ta.technical_data.keys():
        if ticker not in example_tickers and len(example_tickers) < 3:
            example_tickers.append(ticker)
            break

    for ticker in example_tickers:
        try:
            ta.analyze_indicators(ticker)
        except Exception as e:
            print(f"❌ {ticker} analizi hatası: {e}\n")

    input("▶️  Görselleştirmeye geçmek için ENTER'a basın...")

    # ===== 5. GÖRSELLEŞTİRMELER =====
    print("\n" + "=" * 70)
    print("5️⃣  TEKNIK ANALİZ GRAFİKLERİ")
    print("=" * 70 + "\n")

    output_dir = Path('outputs/technical_charts')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Grafikler '{output_dir}' klasörüne kaydedilecek\n")

    graph_count = 0

    # 5.1. Birinci hisse - Fiyat + SMA + RSI + MACD
    if example_tickers and example_tickers[0] in ta.technical_data:
        ticker = example_tickers[0]
        print(f"📈 {ticker} - Fiyat + SMA + RSI + MACD grafiği oluşturuluyor...")
        try:
            df = ta.technical_data[ticker].tail(252)
            fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

            axes[0].plot(df.index, df['close'], label='Kapanış', linewidth=2, color='#2E86AB')
            axes[0].plot(df.index, df['sma_20'], label='SMA(20)', linewidth=1.5, color='#A23B72')
            axes[0].plot(df.index, df['sma_50'], label='SMA(50)', linewidth=1.5, color='#F18F01')
            axes[0].fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1, color='#06A77D')
            axes[0].set_title(f'{ticker} - Fiyat & SMA & Bollinger', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Fiyat', fontsize=12)
            axes[0].legend(loc='best')
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(df.index, df['rsi_14'], label='RSI(14)', linewidth=2, color='#2E86AB')
            axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
            axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
            axes[1].fill_between(df.index, 30, 70, alpha=0.1, color='gray')
            axes[1].set_title('RSI(14)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('RSI', fontsize=11)
            axes[1].set_ylim(0, 100)
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(df.index, df['macd'], label='MACD', linewidth=2, color='#2E86AB')
            axes[2].plot(df.index, df['macd_signal'], label='Signal', linewidth=2, color='#A23B72')
            axes[2].bar(df.index, df['macd_hist'], label='Histogram', color='#06A77D', alpha=0.3)
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[2].set_title('MACD', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Tarih', fontsize=11)
            axes[2].set_ylabel('MACD', fontsize=11)
            axes[2].legend(loc='best')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f'{ticker}_technical_indicators.png', dpi=300, bbox_inches='tight')
            plt.close()
            graph_count += 1
            print(f"✅ Grafik kaydedildi\n")
        except Exception as e:
            print(f"⚠️  Grafik hatası: {e}\n")

    # 5.2. İkinci hisse - Stochastic + Williams %R
    if len(example_tickers) > 1 and example_tickers[1] in ta.technical_data:
        ticker = example_tickers[1]
        print(f"📊 {ticker} - Stochastic + Williams %R grafiği oluşturuluyor...")
        try:
            df = ta.technical_data[ticker].tail(252)
            fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

            axes[0].plot(df.index, df['stochastic_k'], label='K%', linewidth=2, color='#2E86AB')
            axes[0].plot(df.index, df['stochastic_d'], label='D%', linewidth=2, color='#A23B72')
            axes[0].axhline(y=80, color='red', linestyle='--', alpha=0.5)
            axes[0].axhline(y=20, color='green', linestyle='--', alpha=0.5)
            axes[0].fill_between(df.index, 20, 80, alpha=0.1, color='gray')
            axes[0].set_title(f'{ticker} - Stochastic', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Stochastic', fontsize=11)
            axes[0].set_ylim(0, 100)
            axes[0].legend(loc='best')
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(df.index, df['williams_r'], label='Williams %R', linewidth=2, color='#F18F01')
            axes[1].axhline(y=-20, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(y=-80, color='green', linestyle='--', alpha=0.5)
            axes[1].fill_between(df.index, -20, -80, alpha=0.1, color='gray')
            axes[1].set_title('Williams %R', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Tarih', fontsize=11)
            axes[1].set_ylabel('Williams %R', fontsize=11)
            axes[1].set_ylim(-100, 0)
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f'{ticker}_stochastic_williams.png', dpi=300, bbox_inches='tight')
            plt.close()
            graph_count += 1
            print(f"✅ Grafik kaydedildi\n")
        except Exception as e:
            print(f"⚠️  Grafik hatası: {e}\n")

    # 5.3. Üçüncü hisse - Bollinger Bands + ATR
    if len(example_tickers) > 2 and example_tickers[2] in ta.technical_data:
        ticker = example_tickers[2]
        print(f"📈 {ticker} - Bollinger Bands + ATR grafiği oluşturuluyor...")
        try:
            df = ta.technical_data[ticker].tail(252)
            fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

            axes[0].plot(df.index, df['close'], label='Kapanış', linewidth=2, color='#2E86AB')
            axes[0].plot(df.index, df['bb_upper'], label='Upper', linewidth=1.5, color='#A23B72', linestyle='--')
            axes[0].plot(df.index, df['bb_middle'], label='Middle', linewidth=1.5, color='#F18F01')
            axes[0].plot(df.index, df['bb_lower'], label='Lower', linewidth=1.5, color='#06A77D', linestyle='--')
            axes[0].fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
            axes[0].set_title(f'{ticker} - Bollinger Bands', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Fiyat', fontsize=11)
            axes[0].legend(loc='best')
            axes[0].grid(True, alpha=0.3)

            axes[1].bar(df.index, df['atr_14'], label='ATR(14)', color='#06A77D', alpha=0.7)
            axes[1].set_title('ATR', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Tarih', fontsize=11)
            axes[1].set_ylabel('ATR', fontsize=11)
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f'{ticker}_bollinger_atr.png', dpi=300, bbox_inches='tight')
            plt.close()
            graph_count += 1
            print(f"✅ Grafik kaydedildi\n")
        except Exception as e:
            print(f"⚠️  Grafik hatası: {e}\n")

    # 5.4. Sinyal Özeti Tablosu
    print("📊 Sinyal Özeti tablosu oluşturuluyor...")
    try:
        signals_data = []
        for ticker, df in ta.technical_data.items():
            signals_data.append({
                'Ticker': ticker,
                'Signal': df['signal'].iloc[-1],
                'Strength': f"{df['signal_strength'].iloc[-1]:.1%}",
                'RSI': f"{df['rsi_14'].iloc[-1]:.0f}",
                'MACD': '↑' if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else '↓',
                'MFI': f"{df['mfi_14'].iloc[-1]:.0f}"
            })

        signals_df = pd.DataFrame(signals_data)
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=signals_df.values, colLabels=signals_df.columns,
                        cellLoc='center', loc='center', colWidths=[0.15, 0.12, 0.12, 0.1, 0.1, 0.1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        for i in range(len(signals_df)):
            signal = signals_df.iloc[i]['Signal']
            if signal == 'BUY':
                table[(i+1, 1)].set_facecolor('#90EE90')
            elif signal == 'SELL':
                table[(i+1, 1)].set_facecolor('#FFB6C6')
            else:
                table[(i+1, 1)].set_facecolor('#FFFFE0')

        plt.title('Teknik Analiz - Sinyal Özeti', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(output_dir / 'signal_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        graph_count += 1
        print(f"✅ Tablo kaydedildi\n")
    except Exception as e:
        print(f"⚠️  Tablo hatası: {e}\n")

    input("▶️  Devam etmek için ENTER'a basın...")

    # ===== 6. VERİLERİ KAYDET =====
    print("\n" + "=" * 70)
    print("6️⃣  TEKNİK VERİLERİ KAYDETME")
    print("=" * 70 + "\n")

    try:
        ta.save_technical_data(output_dir='data/technical')
    except Exception as e:
        print(f"❌ Veri kaydetme hatası: {e}\n")

    input("▶️  Devam etmek için ENTER'a basın...")

    # ===== 7. KAPSAMLI ÖZET RAPOR =====
    print("\n" + "=" * 70)
    print("7️⃣  KAPSAMLI TEKNİK ANALİZ RAPORU")
    print("=" * 70 + "\n")

    try:
        buy_count = sum(1 for df in ta.technical_data.values() if df['signal'].iloc[-1] == 'BUY')
        sell_count = sum(1 for df in ta.technical_data.values() if df['signal'].iloc[-1] == 'SELL')
        hold_count = sum(1 for df in ta.technical_data.values() if df['signal'].iloc[-1] == 'HOLD')

        print("📊 SINYAL İSTATİSTİKLERİ:")
        print(f"   • BUY Sinyalleri: {buy_count}")
        print(f"   • SELL Sinyalleri: {sell_count}")
        print(f"   • HOLD Sinyalleri: {hold_count}")
        print(f"   • Toplam: {buy_count + sell_count + hold_count}\n")

        avg_rsi = np.mean([df['rsi_14'].iloc[-1] for df in ta.technical_data.values() if not pd.isna(df['rsi_14'].iloc[-1])])
        print(f"📊 ORTALAMA METRİKLER:")
        print(f"   • Ort. RSI(14): {avg_rsi:.2f}\n")

        print(f"📊 EN GÜÇLÜ BUY SİNYALLERİ:")
        buy_signals = [(ticker, df['signal_strength'].iloc[-1])
                       for ticker, df in ta.technical_data.items()
                       if df['signal'].iloc[-1] == 'BUY']
        buy_signals.sort(key=lambda x: x[1], reverse=True)
        for ticker, strength in buy_signals[:5]:
            print(f"   • {ticker:12s} → {strength:.1%}")

        print(f"\n📊 EN GÜÇLÜ SELL SİNYALLERİ:")
        sell_signals = [(ticker, df['signal_strength'].iloc[-1])
                        for ticker, df in ta.technical_data.items()
                        if df['signal'].iloc[-1] == 'SELL']
        sell_signals.sort(key=lambda x: x[1], reverse=True)
        for ticker, strength in sell_signals[:5]:
            print(f"   • {ticker:12s} → {strength:.1%}")

        save_summary_report(ta)

    except Exception as e:
        print(f"❌ Rapor hatası: {e}\n")

    # ===== FINAL ÖZET =====
    print("\n" + "=" * 70)
    print("✨ TEKNİK ANALİZ SÜRECİ TAMAMLANDI!")
    print("=" * 70)
    print(f"📁 Grafikler: {output_dir}")
    print(f"📊 Toplam: {graph_count} grafik oluşturuldu")
    print(f"💾 Teknik veriler: data/technical/ klasörü")
    print(f"📄 Özet rapor: outputs/technical_summary_report.txt")
    print("=" * 70)
    print("\n🎯 Sonraki adım: Makine Öğrenmesi Modelleri - 8-9. Hafta")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {str(e)}")
        import traceback
        traceback.print_exc()