"""
Borsa Trend Analizi - Proje Yapısı Düzenleyici
Ana dizindeki test ve model dosyalarını uygun klasörlere taşır
"""

import os
import shutil
from pathlib import Path


def organize_project_structure():
    """
    Ana dizindeki dosyaları organize eder:
    - test_*.py dosyalarını tests/ klasörüne
    - train_*.py ve diğer script dosyalarını scripts/ klasörüne taşır
    """

    print("=" * 70)
    print("🔧 PROJE YAPISINI DÜZENLİYOR")
    print("=" * 70 + "\n")

    # Ana dizin
    root_dir = Path('.')

    # Hedef klasörler
    scripts_dir = Path('scripts')
    tests_dir = Path('tests')

    # Klasörleri oluştur
    scripts_dir.mkdir(exist_ok=True)
    tests_dir.mkdir(exist_ok=True)

    print("📁 Hedef klasörler hazır:\n")
    print(f"   ✅ {scripts_dir}/")
    print(f"   ✅ {tests_dir}/\n")

    # Taşınacak dosyaları tanımla
    files_to_move = {
        # Test dosyaları
        'test_model.py': tests_dir / 'test_models.py',
        'test_models.py': tests_dir / 'test_models.py',
        'test_data.py': tests_dir / 'test_data_collector.py',
        'test_indicators.py': tests_dir / 'test_indicators.py',
        'test_*.py': tests_dir,  # Wildcard pattern

        # Script dosyaları
        'train_model.py': scripts_dir / 'train_models.py',
        'train_models.py': scripts_dir / 'train_models.py',
        'backtest.py': scripts_dir / 'backtest.py',
        'run_technical.py': scripts_dir / 'run_technical_analysis.py',
        'run_technical_analysis.py': scripts_dir / 'run_technical_analysis.py',
    }

    moved_files = []
    skipped_files = []

    print("🔍 Ana dizin taranıyor...\n")

    # Wildcard olmayan dosyalar için direkt taşıma
    for source_name, target_path in files_to_move.items():
        if '*' in source_name:
            continue

        source_path = root_dir / source_name

        if source_path.exists() and source_path.is_file():
            try:
                # Hedef klasörü al
                if isinstance(target_path, Path) and target_path.is_dir():
                    target_file = target_path / source_name
                else:
                    target_file = target_path

                # Dosyayı taşı
                shutil.move(str(source_path), str(target_file))
                moved_files.append((source_name, target_file))
                print(f"   ✅ {source_name:30s} → {target_file}")

            except Exception as e:
                print(f"   ❌ {source_name}: {str(e)}")
                skipped_files.append((source_name, str(e)))

    # Wildcard pattern için test_*.py dosyalarını tara
    print("\n🔍 test_*.py dosyaları aranıyor...\n")
    for file_path in root_dir.glob('test_*.py'):
        if file_path.name not in ['test_model.py', 'test_models.py', 'test_data.py']:
            try:
                target_file = tests_dir / file_path.name
                shutil.move(str(file_path), str(target_file))
                moved_files.append((file_path.name, target_file))
                print(f"   ✅ {file_path.name:30s} → {target_file}")
            except Exception as e:
                print(f"   ❌ {file_path.name}: {str(e)}")
                skipped_files.append((file_path.name, str(e)))

    # __init__.py dosyalarını oluştur
    print("\n📝 __init__.py dosyaları oluşturuluyor...\n")

    init_files = [
        tests_dir / '__init__.py',
        scripts_dir / '__init__.py',
    ]

    for init_file in init_files:
        if not init_file.exists():
            init_file.touch()
            print(f"   ✅ {init_file}")
        else:
            print(f"   ⚠️  {init_file} (zaten var)")

    # Ana dizinde kalması gereken dosyalar
    print("\n✅ Ana dizinde kalacak dosyalar:\n")

    keep_in_root = [
        'main.py',
        'run_eda.py',
        'app.py',
        'setup_project.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        'LICENSE'
    ]

    for file_name in keep_in_root:
        file_path = root_dir / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ⚠️  {file_name} (bulunamadı)")

    # Özet
    print("\n" + "=" * 70)
    print("📊 DÜZENLEME ÖZET")
    print("=" * 70 + "\n")

    print(f"✅ Taşınan dosyalar: {len(moved_files)}")
    for source, target in moved_files:
        print(f"   • {source} → {target}")

    if skipped_files:
        print(f"\n⚠️  Atlanılan dosyalar: {len(skipped_files)}")
        for source, error in skipped_files:
            print(f"   • {source}: {error}")

    print("\n" + "=" * 70)
    print("✨ PROJE YAPISI DÜZENLENDİ!")
    print("=" * 70)

    print("\n🎯 Sonraki Adımlar:")
    print("   1. Ana dizin temiz ve düzenli")
    print("   2. Test dosyaları tests/ klasöründe")
    print("   3. Script dosyaları scripts/ klasöründe")
    print("   4. Import yollarını kontrol edin")

    print("\n💡 Kullanım Örnekleri:")
    print("   • Testler: pytest tests/")
    print("   • Model eğitimi: python scripts/train_models.py")
    print("   • Backtesting: python scripts/backtest.py")

    print("\n" + "=" * 70 + "\n")

    # Import yolu uyarısı
    if moved_files:
        print("⚠️  ÖNEMLİ: Import yollarını güncelleyin!")
        print("\nÖrnek:")
        print("  # Eskiden:")
        print("  python test_model.py")
        print("\n  # Yeni:")
        print("  pytest tests/test_models.py")
        print("\n  veya")
        print("  python -m pytest tests/test_models.py")
        print()


def check_project_structure():
    """Mevcut proje yapısını kontrol et"""

    print("\n" + "=" * 70)
    print("🔍 MEVCUT PROJE YAPISI KONTROLÜ")
    print("=" * 70 + "\n")

    root_dir = Path('.')

    # Ana dizindeki Python dosyalarını listele
    print("📄 Ana dizindeki Python dosyaları:\n")

    python_files = list(root_dir.glob('*.py'))

    if python_files:
        for py_file in sorted(python_files):
            size_kb = py_file.stat().st_size / 1024
            print(f"   • {py_file.name:30s} ({size_kb:.1f} KB)")
    else:
        print("   (Python dosyası bulunamadı)")

    # Klasörleri kontrol et
    print("\n📁 Klasörler:\n")

    expected_dirs = {
        'data': 'Veri dosyaları',
        'src': 'Kaynak kodlar',
        'scripts': 'Kullanıcı scriptleri',
        'tests': 'Test dosyaları',
        'outputs': 'Çıktı dosyaları',
        'notebooks': 'Jupyter notebooks',
        'streamlit_app': 'Streamlit uygulaması',
        'docs': 'Dokümantasyon'
    }

    for dir_name, description in expected_dirs.items():
        dir_path = root_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            file_count = len(list(dir_path.rglob('*')))
            print(f"   ✅ {dir_name:20s} - {description:30s} ({file_count} öğe)")
        else:
            print(f"   ❌ {dir_name:20s} - {description:30s} (bulunamadı)")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 BORSA TREND ANALİZİ - PROJE YAPISINI DÜZENLE")
    print("=" * 70)

    # Önce mevcut yapıyı kontrol et
    check_project_structure()

    # Onay al
    print("\n⚠️  Bu işlem ana dizindeki dosyaları taşıyacak!")
    response = input("Devam etmek istiyor musunuz? (e/h): ").lower().strip()

    if response in ['e', 'evet', 'y', 'yes']:
        organize_project_structure()
    else:
        print("\n❌ İşlem iptal edildi.")
        print("=" * 70 + "\n")