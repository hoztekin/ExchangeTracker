"""
LazyPredict Sonuç Analizi ve Rapor Üretici
Otomatik olarak en iyi modelleri seçer ve detaylı rapor oluşturur
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class LazyPredictAnalyzer:
    """LazyPredict sonuçlarını otomatik analiz eder"""

    def __init__(self, results_dir='outputs/lazy_predict'):
        self.results_dir = Path(results_dir)
        self.results = {}
        self.analysis = {}

    def load_results(self):
        """Tüm CSV sonuçlarını yükle"""
        print("📂 Sonuçlar yükleniyor...\n")

        csv_files = list(self.results_dir.glob('*_results.csv'))

        if not csv_files:
            print(f"❌ '{self.results_dir}' klasöründe sonuç bulunamadı!")
            return False

        for csv_file in csv_files:
            # Dosya adından ticker ve task'ı çıkar
            filename = csv_file.stem.replace('_results', '')
            parts = filename.split('_')

            if 'classification' in filename:
                ticker = '_'.join(parts[:-1])
                task = 'classification'
            else:  # regression
                ticker = '_'.join(parts[:-1])
                task = 'regression'

            # Veriyi yükle
            df = pd.read_csv(csv_file, index_col=0)

            key = f"{ticker}_{task}"
            self.results[key] = df

            print(f"✅ {key:40s} → {len(df)} model")

        print(f"\n📊 Toplam {len(self.results)} sonuç yüklendi\n")
        return True

    def analyze_classification(self, ticker, df):
        """Classification sonuçlarını analiz et"""

        # En iyi 5 modeli F1 Score'a göre seç
        top_5 = df.sort_values('F1 Score', ascending=False).head(5)

        # Genel istatistikler
        stats = {
            'ticker': ticker,
            'task': 'classification',
            'best_model': top_5.index[0],
            'best_f1': top_5['F1 Score'].iloc[0],
            'best_accuracy': top_5['Accuracy'].iloc[0],
            'best_balanced_acc': top_5['Balanced Accuracy'].iloc[0],
            'avg_f1': df['F1 Score'].mean(),
            'avg_accuracy': df['Accuracy'].mean(),
            'avg_balanced_acc': df['Balanced Accuracy'].mean(),
            'top_5_models': top_5,
            'total_models': len(df)
        }

        # Sorun tespiti
        problems = []
        recommendations = []

        if stats['best_balanced_acc'] < 0.5:
            problems.append("❌ Balanced Accuracy çok düşük - Class imbalance var!")
            recommendations.append("SMOTE kullanarak class balancing yap")
            recommendations.append("Threshold'u değiştir (±2% yerine ±1%)")

        if stats['best_f1'] < 0.70:
            problems.append("⚠️  F1 Score düşük - Model öğrenemiyor")
            recommendations.append("Daha fazla feature ekle (feature engineering)")
            recommendations.append("HOLD class'ını çıkar, sadece BUY/SELL yap")

        # DummyClassifier kontrolü
        if 'DummyClassifier' in df.index:
            dummy_acc = df.loc['DummyClassifier', 'Accuracy']
            if stats['best_accuracy'] - dummy_acc < 0.05:
                problems.append("🚨 Modeller DummyClassifier'dan sadece %5 iyi - Ciddi problem!")
                recommendations.append("Veri kalitesini kontrol et")
                recommendations.append("Feature'ları yeniden düşün")

        stats['problems'] = problems
        stats['recommendations'] = recommendations

        return stats

    def analyze_regression(self, ticker, df):
        """Regression sonuçlarını analiz et"""

        # En iyi 5 modeli R²'ye göre seç
        top_5 = df.sort_values('R-Squared', ascending=False).head(5)

        # Genel istatistikler
        stats = {
            'ticker': ticker,
            'task': 'regression',
            'best_model': top_5.index[0],
            'best_r2': top_5['R-Squared'].iloc[0],
            'best_rmse': top_5['RMSE'].iloc[0],
            'avg_r2': df['R-Squared'].mean(),
            'avg_rmse': df['RMSE'].mean(),
            'top_5_models': top_5,
            'total_models': len(df)
        }

        # Performans değerlendirmesi
        problems = []
        recommendations = []
        grade = None

        if stats['best_r2'] >= 0.90:
            grade = "🏆 MÜKEMMEL"
            recommendations.append("Model production'a hazır!")
            recommendations.append("Hiperparametre tuning ile daha da iyileştir")
        elif stats['best_r2'] >= 0.80:
            grade = "✅ ÇOK İYİ"
            recommendations.append("Model kullanılabilir")
            recommendations.append("Walk-forward validation yap")
        elif stats['best_r2'] >= 0.70:
            grade = "⚠️  ORTA"
            problems.append("R² biraz düşük")
            recommendations.append("Daha fazla feature ekle")
            recommendations.append("Feature engineering yap")
        else:
            grade = "❌ ZAYIF"
            problems.append("R² çok düşük - Model öğrenemiyor")
            recommendations.append("Veri kalitesini kontrol et")
            recommendations.append("Farklı modeller dene (LSTM, etc.)")

        stats['grade'] = grade
        stats['problems'] = problems
        stats['recommendations'] = recommendations

        # Şaşırtıcı sonuçları bul
        surprising = []

        # XGBoost/LightGBM/RandomForest beklenenden kötüyse
        for model in ['XGBRegressor', 'LGBMRegressor', 'RandomForestRegressor']:
            if model in df.index:
                model_r2 = df.loc[model, 'R-Squared']
                if model_r2 < stats['best_r2'] - 0.2:
                    surprising.append(f"⚠️  {model} beklenenden kötü (R²: {model_r2:.3f})")

        # Linear modeller beklenenden iyiyse
        linear_models = ['Ridge', 'LinearRegression', 'Lasso']
        for model in linear_models:
            if model in df.index:
                model_r2 = df.loc[model, 'R-Squared']
                if model == stats['best_model']:
                    surprising.append(f"💡 {model} en iyi model - Linear modeller finansal veri için uygun!")

        stats['surprising'] = surprising

        return stats

    def analyze_all(self):
        """Tüm sonuçları analiz et"""
        print("=" * 70)
        print("🔍 SONUÇLAR ANALİZ EDİLİYOR")
        print("=" * 70 + "\n")

        for key, df in self.results.items():
            ticker = '_'.join(key.split('_')[:-1])

            if 'classification' in key:
                analysis = self.analyze_classification(ticker, df)
            else:
                analysis = self.analyze_regression(ticker, df)

            self.analysis[key] = analysis
            print(f"✅ {key} analizi tamamlandı")

        print(f"\n📊 Toplam {len(self.analysis)} analiz tamamlandı\n")

    def generate_report(self, output_file='outputs/lazy_predict_analysis.txt'):
        """Detaylı rapor oluştur"""

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 70 + "\n")
            f.write("📊 LAZYPREDICT OTOMATIK ANALİZ RAPORU\n")
            f.write("=" * 70 + "\n")
            f.write(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"📂 Veri: {self.results_dir}\n")
            f.write(f"📊 Analiz Sayısı: {len(self.analysis)}\n")
            f.write("=" * 70 + "\n\n")

            # Executive Summary
            f.write("━" * 70 + "\n")
            f.write("🎯 YÖNETİCİ ÖZETİ\n")
            f.write("━" * 70 + "\n\n")

            # Classification özeti
            clf_analyses = {k: v for k, v in self.analysis.items() if 'classification' in k}
            if clf_analyses:
                avg_f1 = np.mean([a['best_f1'] for a in clf_analyses.values()])
                avg_bal_acc = np.mean([a['best_balanced_acc'] for a in clf_analyses.values()])

                f.write("📊 CLASSIFICATION PERFORMANSI:\n")
                f.write(f"   • Ortalama F1 Score: {avg_f1:.3f}\n")
                f.write(f"   • Ortalama Balanced Accuracy: {avg_bal_acc:.3f}\n")

                if avg_f1 < 0.70:
                    f.write("   • Durum: ⚠️  ZAYIF - İyileştirme gerekli\n")
                else:
                    f.write("   • Durum: ✅ İYİ\n")
                f.write("\n")

            # Regression özeti
            reg_analyses = {k: v for k, v in self.analysis.items() if 'regression' in k}
            if reg_analyses:
                avg_r2 = np.mean([a['best_r2'] for a in reg_analyses.values()])

                f.write("📈 REGRESSION PERFORMANSI:\n")
                f.write(f"   • Ortalama R²: {avg_r2:.3f}\n")

                if avg_r2 >= 0.85:
                    f.write("   • Durum: 🏆 MÜKEMMEL - Production'a hazır!\n")
                elif avg_r2 >= 0.70:
                    f.write("   • Durum: ✅ ÇOK İYİ\n")
                else:
                    f.write("   • Durum: ⚠️  İyileştirme gerekli\n")
                f.write("\n")

            # Genel öneri
            f.write("🎯 GENEL ÖNERİ:\n")
            if reg_analyses and np.mean([a['best_r2'] for a in reg_analyses.values()]) >= 0.80:
                f.write("   ✅ Regression modelleri kullanıma hazır!\n")
                f.write("   ✅ Fiyat tahmini için hemen kullanabilirsin\n")
            if clf_analyses and np.mean([a['best_f1'] for a in clf_analyses.values()]) < 0.70:
                f.write("   ⚠️  Classification modelleri iyileştirilmeli\n")
                f.write("   ⚠️  Sinyal üretimi için feature engineering gerekli\n")

            f.write("\n" + "=" * 70 + "\n\n")

            # Detaylı analizler (devamı...)
            for key, analysis in self.analysis.items():
                ticker = analysis['ticker']
                task = analysis['task'].upper()

                f.write("=" * 70 + "\n")
                f.write(f"📊 {ticker} - {task}\n")
                f.write("=" * 70 + "\n\n")

                if task == 'CLASSIFICATION':
                    self._write_classification_section(f, analysis)
                else:
                    self._write_regression_section(f, analysis)

                f.write("\n")

        print(f"✅ Rapor oluşturuldu: {output_file}\n")
        return output_file

    def _write_classification_section(self, f, analysis):
        """Classification bölümünü yaz"""

        f.write("📊 EN İYİ 5 MODEL:\n")
        f.write("─" * 70 + "\n")

        for i, (model_name, row) in enumerate(analysis['top_5_models'].iterrows(), 1):
            f.write(f"{i}. {model_name}\n")
            f.write(
                f"   F1: {row['F1 Score']:.3f} | Acc: {row['Accuracy']:.3f} | Bal.Acc: {row['Balanced Accuracy']:.3f}\n")

        f.write("\n")

        if analysis['problems']:
            f.write("⚠️  SORUNLAR:\n")
            for p in analysis['problems']:
                f.write(f"   {p}\n")
            f.write("\n")

        if analysis['recommendations']:
            f.write("💡 ÖNERİLER:\n")
            for i, r in enumerate(analysis['recommendations'], 1):
                f.write(f"   {i}. {r}\n")

    def _write_regression_section(self, f, analysis):
        """Regression bölümünü yaz"""

        f.write(f"📈 PERFORMANS: {analysis['grade']}\n\n")
        f.write("📊 EN İYİ 5 MODEL:\n")
        f.write("─" * 70 + "\n")

        for i, (model_name, row) in enumerate(analysis['top_5_models'].iterrows(), 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
            f.write(f"{emoji} {i}. {model_name}\n")
            f.write(f"   R²: {row['R-Squared']:.4f} | RMSE: {row['RMSE']:.3f}\n")

        f.write("\n")

        if analysis.get('surprising'):
            f.write("💡 İLGİNÇ:\n")
            for s in analysis['surprising']:
                f.write(f"   {s}\n")
            f.write("\n")

        if analysis['recommendations']:
            f.write("💡 ÖNERİLER:\n")
            for i, r in enumerate(analysis['recommendations'], 1):
                f.write(f"   {i}. {r}\n")

    def print_summary(self):
        """Terminal özeti"""
        print("\n" + "=" * 70)
        print("📊 ÖZET")
        print("=" * 70 + "\n")

        for key, analysis in self.analysis.items():
            print(f"{'━' * 35}")
            print(f"📌 {analysis['ticker']} - {analysis['task'].upper()}")

            if analysis['task'] == 'classification':
                print(f"   🏆 {analysis['best_model']}")
                print(f"   📊 F1: {analysis['best_f1']:.3f}")
            else:
                print(f"   🏆 {analysis['best_model']}")
                print(f"   📊 R²: {analysis['best_r2']:.4f}")
                print(f"   {analysis['grade']}")
            print()


def main():
    """Ana program"""
    print("=" * 70)
    print("🤖 LAZYPREDICT OTOMATIK ANALİZ")
    print("=" * 70 + "\n")

    analyzer = LazyPredictAnalyzer(results_dir='outputs/lazy_predict')

    if not analyzer.load_results():
        return

    analyzer.analyze_all()
    analyzer.print_summary()

    report_path = analyzer.generate_report()

    print("=" * 70)
    print("✅ TAMAMLANDI!")
    print("=" * 70)
    print(f"📄 Rapor: {report_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Hata: {str(e)}")
        import traceback

        traceback.print_exc()