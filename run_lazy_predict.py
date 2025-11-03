"""
LazyPredict ile Model Keşfi - Çalıştırma Scripti
8. Hafta: Otomatik Model Seçimi
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.models.lazy_model_selector import LazyModelSelector
import warnings

warnings.filterwarnings('ignore')


def print_banner(text):
    """Güzel banner yazdır"""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def main():
    print_banner("🚀 LAZYPREDICT - OTOMATİK MODEL KEŞFİ")
    print("8. Hafta: Makine Öğrenmesi - Model Discovery")
    print("Tüm modeller otomatik test edilecek ve en iyileri belirlenecek!\n")

    # Selector'ı başlat
    try:
        selector = LazyModelSelector(data_dir='data/technical')
        print("✅ LazyModelSelector başlatıldı\n")
    except ImportError as e:
        print(f"❌ HATA: {str(e)}")
        print("\n💡 ÇÖZÜM:")
        print("   pip install lazypredict xgboost lightgbm catboost")
        return

    # Test edilecek hisseler
    test_tickers = [
        'THYAO_IS',  # BIST - Türk Hava Yolları
        'AAPL',  # S&P 500 - Apple
        'GARAN_IS',  # BIST - Garanti Bankası
        'MSFT',  # S&P 500 - Microsoft
    ]

    print(f"📊 Test Edilecek Hisseler:")
    for i, ticker in enumerate(test_tickers, 1):
        print(f"   {i}. {ticker}")

    print(f"\n💡 Her hisse için hem Classification hem Regression test edilecek")
    print(f"⏱️  Tahmini süre: ~5-10 dakika (hisse başı)")

    input("\n▶️  Başlamak için ENTER'a basın...")

    # Sonuçları topla
    all_results = {
        'classification': {},
        'regression': {}
    }

    # Her hisse için çalıştır
    for idx, ticker in enumerate(test_tickers, 1):

        print_banner(f"{idx}/{len(test_tickers)} - {ticker}")

        # ===== CLASSIFICATION =====
        print(f"🎯 ADIM 1/2: Classification (Sinyal Tahmini)")
        print(f"   Hedef: Yarın BUY/HOLD/SELL sinyali üret\n")

        try:
            clf_results = selector.run_classification(
                ticker,
                threshold=0.02,  # ±%2 eşik
                test_size=0.2
            )

            if clf_results is not None:
                all_results['classification'][ticker] = clf_results
                print(f"\n✅ {ticker} Classification tamamlandı!")

                # En iyi 3 modeli göster
                top_3 = clf_results.sort_values('Accuracy', ascending=False).head(3)
                print(f"\n🏆 EN İYİ 3 MODEL:")
                for i, (model_name, row) in enumerate(top_3.iterrows(), 1):
                    print(f"   {i}. {model_name:30s} → Accuracy: {row['Accuracy']:.3f}")
            else:
                print(f"\n⚠️  {ticker} Classification başarısız!")

        except Exception as e:
            print(f"\n❌ {ticker} Classification hatası: {str(e)}")

        input(f"\n▶️  {ticker} Regression'a geçmek için ENTER...")

        # ===== REGRESSION =====
        print(f"\n🎯 ADIM 2/2: Regression (Fiyat Tahmini)")
        print(f"   Hedef: Yarının kapanış fiyatını tahmin et\n")

        try:
            reg_results = selector.run_regression(
                ticker,
                test_size=0.2
            )

            if reg_results is not None:
                all_results['regression'][ticker] = reg_results
                print(f"\n✅ {ticker} Regression tamamlandı!")

                # En iyi 3 modeli göster
                top_3 = reg_results.sort_values('R-Squared', ascending=False).head(3)
                print(f"\n🏆 EN İYİ 3 MODEL:")
                for i, (model_name, row) in enumerate(top_3.iterrows(), 1):
                    print(f"   {i}. {model_name:30s} → R²: {row['R-Squared']:.3f}, RMSE: {row['RMSE']:.2f}")
            else:
                print(f"\n⚠️  {ticker} Regression başarısız!")

        except Exception as e:
            print(f"\n❌ {ticker} Regression hatası: {str(e)}")

        # Sonraki hisseye geç
        if idx < len(test_tickers):
            input(f"\n▶️  Sonraki hisse ({test_tickers[idx]}) için ENTER...")

    # ===== SONUÇLARI KAYDET =====
    print_banner("💾 SONUÇLARI KAYDETME")

    try:
        selector.save_results()
        selector.generate_summary_report()
        print("✅ Tüm sonuçlar kaydedildi!")
    except Exception as e:
        print(f"⚠️  Kaydetme hatası: {str(e)}")

    # ===== GENEL ÖZET =====
    print_banner("📊 GENEL ÖZET")

    print("✅ TAMAMLANAN İŞLEMLER:")
    print(f"   • Test edilen hisse sayısı: {len(test_tickers)}")
    print(f"   • Classification başarılı: {len(all_results['classification'])}")
    print(f"   • Regression başarılı: {len(all_results['regression'])}")

    if all_results['classification']:
        print(f"\n🏆 EN İYİ CLASSIFICATION MODELLER (GENEL):")

        # Her hisse için en iyi modeli bul
        best_models = {}
        for ticker, results in all_results['classification'].items():
            best = results.sort_values('Accuracy', ascending=False).iloc[0]
            best_models[ticker] = (best.name, best['Accuracy'])

        for ticker, (model, acc) in sorted(best_models.items(), key=lambda x: x[1][1], reverse=True):
            print(f"   • {ticker:12s} → {model:30s} (Acc: {acc:.3f})")

    if all_results['regression']:
        print(f"\n🏆 EN İYİ REGRESSION MODELLER (GENEL):")

        # Her hisse için en iyi modeli bul
        best_models = {}
        for ticker, results in all_results['regression'].items():
            best = results.sort_values('R-Squared', ascending=False).iloc[0]
            best_models[ticker] = (best.name, best['R-Squared'])

        for ticker, (model, r2) in sorted(best_models.items(), key=lambda x: x[1][1], reverse=True):
            print(f"   • {ticker:12s} → {model:30s} (R²: {r2:.3f})")

    # ===== SONRAKİ ADIMLAR =====
    print_banner("🎯 SONRAKİ ADIMLAR")

    print("1. 📊 Sonuçları İncele:")
    print("   • outputs/lazy_predict/ klasöründeki CSV'leri aç")
    print("   • summary_report.txt dosyasını oku")

    print("\n2. 🎯 En İyi Modelleri Seç:")
    print("   • Classification: XGBoost, LightGBM, RandomForest")
    print("   • Regression: XGBoost, GradientBoosting, ExtraTrees")

    print("\n3. 🔧 Hiperparametre Tuning:")
    print("   • Seçilen modelleri GridSearchCV ile optimize et")
    print("   • Walk-forward validation kullan")

    print("\n4. 💰 Backtesting:")
    print("   • Gerçek trading simülasyonu yap")
    print("   • Sharpe Ratio, Max Drawdown hesapla")

    print("\n5. 🚀 Production:")
    print("   • En iyi modeli kaydet (.pkl)")
    print("   • Streamlit app'e entegre et")

    print_banner("✨ LAZYPREDICT TAMAMLANDI!")

    print("🎉 Harika! Artık hangi modellerin işe yaradığını biliyorsun!")
    print("🚀 Sırada: En iyi modelleri optimize etme zamanı!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {str(e)}")
        import traceback

        traceback.print_exc()