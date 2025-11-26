"""
Borsa Trend Analizi - EDA Runner
3-4. Hafta: Keşifsel Veri Analizi Çalıştırıcı
"""

import sys
import os
from pathlib import Path

# src klasörünü path'e ekle
sys.path.append(str(Path(__file__).parent))

from src.analysis.eda import ExploratoryDataAnalysis
from src.utils.visualization import StockVisualizer
import warnings
warnings.filterwarnings('ignore')


def main():
    """Ana EDA işlem akışı"""

    print("="*70)
    print("📊 BORSA TREND ANALİZİ - KEŞİFSEL VERİ ANALİZİ (EDA)")
    print("="*70)
    print("3-4. Hafta: Exploratory Data Analysis")
    print("="*70 + "\n")

    # EDA ve Visualizer başlat
    eda = ExploratoryDataAnalysis(data_dir='data/raw')
    viz = StockVisualizer()

    # 1. VERİLERİ YÜKLE
    print("="*70)
    print("1️⃣  VERİ YÜKLEME")
    print("="*70 + "\n")

    eda.load_data()

    if not eda.data:
        print("❌ Veri yüklenemedi! Önce main.py'yi çalıştırın.")
        return

    input("\n▶️  Devam etmek için ENTER'a basın...")

    # 2. TEMEL İSTATİSTİKLER
    print("\n" + "="*70)
    print("2️⃣  TEMEL İSTATİSTİKLER")
    print("="*70 + "\n")

    stats_df = eda.calculate_basic_stats()
    print(stats_df.to_string(index=False))

    input("\n▶️  Devam etmek için ENTER'a basın...")

    # 3. PİYASA KARŞILAŞTIRMASI
    print("\n" + "="*70)
    print("3️⃣  PİYASA KARŞILAŞTIRMASI")
    print("="*70 + "\n")

    eda.compare_markets()

    input("\n▶️  Devam etmek için ENTER'a basın...")

    # 4. DETAYLI ANALİZ - BIST ÖRNEĞİ
    print("\n" + "="*70)
    print("4️⃣  DETAYLI ANALİZ - ÖRNEKler")
    print("="*70 + "\n")

    # BIST örneği - Türk Hava Yolları
    if 'THYAO_IS' in eda.data:
        eda.analyze_price_movements('THYAO_IS')
        eda.detect_seasonal_patterns('THYAO_IS')
        eda.analyze_volume_price_relationship('THYAO_IS')

    # S&P 500 örneği - Apple
    if 'AAPL' in eda.data:
        eda.analyze_price_movements('AAPL')
        eda.detect_seasonal_patterns('AAPL')

    input("\n▶️  Görselleştirmeye geçmek için ENTER'a basın...")

    # 5. GÖRSELLEŞTİRMELER
    print("\n" + "="*70)
    print("5️⃣  GÖRSELLEŞTİRMELER")
    print("="*70 + "\n")

    # Grafik klasörü oluştur
    output_dir = Path('outputs/eda_charts')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Grafikler '{output_dir}' klasörüne kaydedilecek\n")

    # 5.1. Fiyat Geçmişi
    if 'THYAO_IS' in eda.data:
        print("📈 THYAO.IS - Fiyat Geçmişi grafiği oluşturuluyor...")
        viz.plot_price_history(
            eda.data['THYAO_IS'],
            'THYAO.IS',
            save_path=output_dir / 'THYAO_price_history.png'
        )

    if 'AAPL' in eda.data:
        print("📈 AAPL - Fiyat Geçmişi grafiği oluşturuluyor...")
        viz.plot_price_history(
            eda.data['AAPL'],
            'AAPL',
            save_path=output_dir / 'AAPL_price_history.png'
        )

    # 5.2. Candlestick
    if 'MSFT' in eda.data:
        print("🕯️  MSFT - Candlestick grafiği oluşturuluyor...")
        viz.plot_candlestick(
            eda.data['MSFT'],
            'MSFT',
            period=90,
            save_path=output_dir / 'MSFT_candlestick.png'
        )

    # 5.3. Getiri Dağılımı
    if 'TSLA' in eda.data:
        print("📊 TSLA - Getiri Dağılımı grafiği oluşturuluyor...")
        viz.plot_returns_distribution(
            eda.data['TSLA'],
            'TSLA',
            save_path=output_dir / 'TSLA_returns_dist.png'
        )

    # 5.4. Korelasyon Matrisi - BIST
    print("\n🔗 BIST - Korelasyon matrisi oluşturuluyor...")
    corr_bist = eda.calculate_correlation_matrix(market='bist')
    viz.plot_correlation_heatmap(
        corr_bist,
        title='BIST Hisseleri - Korelasyon Matrisi',
        save_path=output_dir / 'correlation_bist.png'
    )

    # 5.5. Korelasyon Matrisi - S&P 500
    print("🔗 S&P 500 - Korelasyon matrisi oluşturuluyor...")
    corr_sp500 = eda.calculate_correlation_matrix(market='sp500')
    viz.plot_correlation_heatmap(
        corr_sp500,
        title='S&P 500 Hisseleri - Korelasyon Matrisi',
        save_path=output_dir / 'correlation_sp500.png'
    )

    # 5.6. Volatilite Karşılaştırması
    print("📊 Volatilite karşılaştırması oluşturuluyor...")
    viz.plot_volatility_comparison(
        eda.data,
        save_path=output_dir / 'volatility_comparison.png'
    )

    # 5.7. Kümülatif Getiri - BIST
    bist_tickers = [t for t in eda.data.keys() if '_IS' in t][:5]
    if bist_tickers:
        print(f"📈 BIST - Kümülatif getiri ({len(bist_tickers)} hisse) oluşturuluyor...")
        viz.plot_cumulative_returns(
            eda.data,
            bist_tickers,
            save_path=output_dir / 'cumulative_returns_bist.png'
        )

    # 5.8. Kümülatif Getiri - S&P 500
    sp500_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    available_sp500 = [t for t in sp500_tickers if t in eda.data]
    if available_sp500:
        print(f"📈 S&P 500 - Kümülatif getiri ({len(available_sp500)} hisse) oluşturuluyor...")
        viz.plot_cumulative_returns(
            eda.data,
            available_sp500,
            save_path=output_dir / 'cumulative_returns_sp500.png'
        )

    # 5.9. Risk-Return Analizi
    print("⚖️  Risk-Return scatter plot oluşturuluyor...")
    viz.plot_risk_return_scatter(
        eda.data,
        save_path=output_dir / 'risk_return_analysis.png'
    )

    # 5.10. Mevsimsel Paternler
    if 'THYAO_IS' in eda.data:
        print("📅 THYAO.IS - Mevsimsel paternler oluşturuluyor...")
        viz.plot_seasonal_patterns(
            eda.data['THYAO_IS'],
            'THYAO.IS',
            save_path=output_dir / 'seasonal_THYAO.png'
        )

    if 'NVDA' in eda.data:
        print("📅 NVDA - Mevsimsel paternler oluşturuluyor...")
        viz.plot_seasonal_patterns(
            eda.data['NVDA'],
            'NVDA',
            save_path=output_dir / 'seasonal_NVDA.png'
        )

    # 5.11. Volume-Price İlişkisi
    if 'AKBNK_IS' in eda.data:
        print("📦 AKBNK.IS - Volume-Price ilişkisi oluşturuluyor...")
        viz.plot_volume_price_relationship(
            eda.data['AKBNK_IS'],
            'AKBNK.IS',
            save_path=output_dir / 'volume_price_AKBNK.png'
        )

    input("\n▶️  Devam etmek için ENTER'a basın...")

    # 6. KAPSAMLI ÖZET RAPOR
    print("\n" + "="*70)
    print("6️⃣  KAPSAMLI ÖZET RAPOR")
    print("="*70 + "\n")

    eda.generate_summary_report()

    # SONUÇ
    print("\n" + "="*70)
    print("✨ EDA SÜRECİ TAMAMLANDI!")
    print("="*70)
    print(f"📁 Grafikler: {output_dir}")
    print(f"📊 Toplam: {len(list(output_dir.glob('*.png')))} görsel oluşturuldu")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()