"""
Pipeline Configuration
Tüm pipeline ayarları burada
"""
import os
from datetime import time

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'technical')  # Technical klasörünü kullan
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
STATE_FILE = os.path.join(BASE_DIR, 'pipeline_state.json')

# Scheduler Settings
SCHEDULER_ENABLED = True
UPDATE_TIME = time(2, 0)  # 02:00
TIMEZONE = 'Europe/Istanbul'

# Model Training Settings
MIN_R2_SCORE = 0.85  # Bu değerin altına düşerse model yeniden eğitilir
RETRAIN_THRESHOLD_DAYS = 7  # Son eğitimden bu kadar gün geçtiyse kontrol et

# Stock Settings
BIST30_STOCKS = [
    'GARAN.IS', 'THYAO.IS', 'AKBNK.IS', 'EREGL.IS', 'TUPRS.IS',
    'KCHOL.IS', 'SAHOL.IS', 'ASELS.IS', 'SISE.IS', 'TCELL.IS'
]

SP500_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'META', 'NVDA', 'JPM', 'V', 'WMT'
]

# Data Settings
DATA_PERIOD = '5y'
DATA_INTERVAL = '1d'

# Technical Indicators
INDICATORS = [
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
    'ATR', 'OBV', 'Stochastic'
]

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'