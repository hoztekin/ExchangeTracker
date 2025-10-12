"""
Proje klasör yapısını otomatik oluşturur
"""

import os
from pathlib import Path


def create_project_structure():
    """README'deki klasör yapısını oluştur"""

    print("=" * 70)
    print("📁 PROJE YAPISINI OLUŞTUR")
    print("=" * 70 + "\n")

    # Klasör yapısı
    folders = [
        # Data
        'data/raw',
        'data/processed',

        # Notebooks
        'notebooks',

        # Source code
        'src/data',
        'src/analysis',
        'src/models',
        'src/utils',

        # Streamlit app
        'streamlit_app/pages',
        'streamlit_app/components',

        # Tests
        'tests',

        # Docs
        'docs',

        # Outputs
        'outputs/eda_charts',
        'outputs/models',
        'outputs/reports'
    ]

    print("📂 Klasörler oluşturuluyor...\n")

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✅ {folder}")

    print("\n📝 __init__.py dosyaları oluşturuluyor...\n")

    # __init__.py dosyaları
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/analysis/__init__.py',
        'src/models/__init__.py',
        'src/utils/__init__.py',
    ]

    for init_file in init_files:
        Path(init_file).touch(exist_ok=True)
        print(f"✅ {init_file}")

    # requirements.txt oluştur
    print("\n📦 requirements.txt oluşturuluyor...")

    requirements = """# Veri İşleme
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0

# Görselleştirme
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Makine Öğrenmesi
scikit-learn>=1.3.0
tensorflow>=2.13.0
statsmodels>=0.14.0

# Teknik Analiz
ta>=0.11.0

# Web Uygulaması
streamlit>=1.28.0

# Utilities
tqdm>=4.65.0
"""

    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)

    print("✅ requirements.txt")

    # .gitignore oluştur
    print("\n🚫 .gitignore oluşturuluyor...")

    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data
data/raw/*.csv
data/processed/*.csv

# Outputs
outputs/
*.png
*.jpg
*.pdf

# Models
*.h5
*.pkl
*.joblib

# OS
.DS_Store
Thumbs.db

# Logs
*.log
"""

    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore)

    print("✅ .gitignore")

    # README içeriğini güncelle
    print("\n📄 README.md güncelleniyor...")

    readme_addition = """
## 🎉 Kurulum Tamamlandı!

Proje yapısı başarıyla oluşturuldu. Şimdi şu adımları takip edin:

### 1. Sanal Ortam Oluşturun (Opsiyonel ama önerilen)
```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

### 2. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Veri Toplayın
```bash
python main.py
```

### 4. EDA Çalıştırın
```bash
python run_eda.py
```

## 📚 Dosya Açıklamaları

- `main.py`: Veri toplama scripti
- `run_eda.py`: Keşifsel veri analizi scripti
- `src/analysis/eda.py`: EDA sınıfı
- `src/utils/visualization.py`: Görselleştirme araçları
- `data/`: CSV veri dosyaları
- `outputs/`: Grafikler ve raporlar
"""

    try:
        with open('README.md', 'a', encoding='utf-8') as f:
            f.write(readme_addition)
        print("✅ README.md güncellendi")
    except:
        print("⚠️  README.md güncellenemedi (devam ediliyor)")

    # Son özet
    print("\n" + "=" * 70)
    print("✨ PROJE YAPISI BAŞARIYLA OLUŞTURULDU!")
    print("=" * 70)
    print("\n📊 Oluşturulan klasörler:")
    print(f"   • {len(folders)} klasör")
    print(f"   • {len(init_files)} __init__.py dosyası")
    print("   • requirements.txt")
    print("   • .gitignore")

    print("\n🎯 Sonraki Adımlar:")
    print("   1. pip install -r requirements.txt")
    print("   2. python main.py  (Veri toplama - TAMAM ✅)")
    print("   3. python run_eda.py  (EDA analizi)")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    create_project_structure()