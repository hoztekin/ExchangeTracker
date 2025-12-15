"""
Pipeline Scheduler Module
Otomatik veri güncelleme ve model eğitimi
"""
import json
import os
import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from .config import (
    STATE_FILE, UPDATE_TIME, TIMEZONE,
    BIST30_STOCKS, SP500_STOCKS, SCHEDULER_ENABLED,
    LOG_FORMAT, LOG_LEVEL
)
from .data_updater import DataUpdater
from .model_trainer import ModelTrainer

# Logging setup
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)


class PipelineScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone=pytz.timezone(TIMEZONE))
        self.state_file = STATE_FILE
        self.data_updater = DataUpdater()
        self.model_trainer = ModelTrainer()

        # State dosyasını oluştur/yükle
        self.load_or_create_state()

    def load_or_create_state(self):
        """State dosyasını yükle veya oluştur"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self.state = json.load(f)
                logger.info(f"✓ State dosyası yüklendi: {self.state_file}")
            except Exception as e:
                logger.error(f"✗ State yükleme hatası: {e}, yeni state oluşturuluyor")
                self._create_initial_state()
        else:
            self._create_initial_state()

    def _create_initial_state(self):
        """İlk state yapısını oluştur"""
        self.state = {
            'last_update': None,
            'next_scheduled': self._get_next_scheduled_time(),
            'status': 'idle',
            'stocks': {}
        }
        self.save_state()
        logger.info(f"✓ Yeni state dosyası oluşturuldu: {self.state_file}")

    def _get_next_scheduled_time(self) -> str:
        """Sonraki otomatik çalışma zamanını hesapla"""
        now = datetime.now()
        next_run = now.replace(hour=UPDATE_TIME.hour, minute=UPDATE_TIME.minute, second=0, microsecond=0)

        if next_run <= now:
            next_run += timedelta(days=1)

        return next_run.strftime('%Y-%m-%d %H:%M:%S')

    def save_state(self):
        """State'i dosyaya kaydet"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ State kaydedildi: {self.state_file}")
        except Exception as e:
            logger.error(f"✗ State kaydetme hatası: {e}")

    def update_stock_state(self, ticker: str, data: dict):
        """Belirli bir hisse için state güncelle"""
        if ticker not in self.state['stocks']:
            self.state['stocks'][ticker] = {}

        self.state['stocks'][ticker].update(data)
        self.save_state()

    def update_all_stocks(self, stock_list: list = None):
        """
        Tüm hisseleri güncelle (otomatik veya manuel)
        """
        if stock_list is None:
            stock_list = BIST30_STOCKS[:3]  # İlk 3 hisse test için

        logger.info(f"🔄 Pipeline başlatıldı: {len(stock_list)} hisse")
        self.state['status'] = 'running'
        self.state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_state()

        results = {
            'data_updates': {},
            'model_updates': {}
        }

        try:
            # 1. Tüm hisseler için veri güncelle
            logger.info("📊 Veri güncelleme aşaması...")
            for ticker in stock_list:
                update_result = self.data_updater.update_stock(ticker)
                results['data_updates'][ticker] = update_result

                # State'e kaydet
                self.update_stock_state(ticker, {
                    'last_data_update': update_result.get('timestamp'),
                    'data_status': update_result.get('status'),
                    'last_date': update_result.get('last_date')
                })

            # 2. Gerekirse modelleri eğit/değerlendir
            logger.info("🤖 Model değerlendirme aşaması...")
            for ticker in stock_list:
                # Sadece veri güncellendiyse veya model yoksa eğit
                data_status = results['data_updates'][ticker].get('status')

                if data_status in ['updated', 'initialized']:
                    model_result = self.model_trainer.train_model(ticker, force_retrain=False)
                    results['model_updates'][ticker] = model_result

                    # State'e kaydet
                    self.update_stock_state(ticker, {
                        'last_model_update': model_result.get('timestamp'),
                        'model_name': model_result.get('model_name'),
                        'r2_score': model_result.get('r2_score'),
                        'model_status': model_result.get('status')
                    })
                else:
                    logger.info(f"ℹ {ticker} için model güncelleme atlandı (veri değişmedi)")

            # Pipeline tamamlandı
            self.state['status'] = 'idle'
            self.state['next_scheduled'] = self._get_next_scheduled_time()
            self.save_state()

            logger.info("✓ Pipeline tamamlandı")
            return results

        except Exception as e:
            logger.error(f"✗ Pipeline hatası: {e}")
            self.state['status'] = 'error'
            self.state['last_error'] = str(e)
            self.save_state()
            return results

    def manual_update_stock(self, ticker: str):
        """Tek bir hisse için manuel güncelleme"""
        logger.info(f"🔄 Manuel güncelleme başlatıldı: {ticker}")
        return self.update_all_stocks([ticker])

    def manual_train_model(self, ticker: str):
        """Tek bir hisse için manuel model eğitimi"""
        logger.info(f"🤖 Manuel model eğitimi başlatıldı: {ticker}")
        self.state['status'] = 'running'
        self.save_state()

        try:
            result = self.model_trainer.train_model(ticker, force_retrain=True)

            # State güncelle
            self.update_stock_state(ticker, {
                'last_model_update': result.get('timestamp'),
                'model_name': result.get('model_name'),
                'r2_score': result.get('r2_score'),
                'model_status': result.get('status')
            })

            self.state['status'] = 'idle'
            self.save_state()

            return result

        except Exception as e:
            logger.error(f"✗ Model eğitim hatası: {e}")
            self.state['status'] = 'error'
            self.save_state()
            return {'status': 'error', 'message': str(e)}

    def start(self):
        """Scheduler'ı başlat"""
        if not SCHEDULER_ENABLED:
            logger.info("⚠ Scheduler devre dışı (config)")
            return

        # Cron job ekle
        trigger = CronTrigger(
            hour=UPDATE_TIME.hour,
            minute=UPDATE_TIME.minute,
            timezone=TIMEZONE
        )

        self.scheduler.add_job(
            func=self.update_all_stocks,
            trigger=trigger,
            id='daily_update',
            name='Daily Stock Update',
            replace_existing=True
        )

        self.scheduler.start()
        logger.info(f"✓ Scheduler başlatıldı: Her gün {UPDATE_TIME.hour:02d}:{UPDATE_TIME.minute:02d}")
        logger.info(f"ℹ Sonraki çalışma: {self.state['next_scheduled']}")

    def stop(self):
        """Scheduler'ı durdur"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("⏸ Scheduler durduruldu")

    def get_state(self) -> dict:
        """Mevcut state'i döndür"""
        return self.state


if __name__ == "__main__":
    # Test
    scheduler = PipelineScheduler()
    print("State:", json.dumps(scheduler.get_state(), indent=2))

    # 1. Veri güncelle
    updater = DataUpdater()
    print("Data update:", updater.update_stock('GARAN.IS'))

    # 2. Modeli eğit
    trainer = ModelTrainer()
    print("Model train:", trainer.train_model('GARAN.IS'))

    # Manuel test
    # result = scheduler.manual_update_stock('GARAN.IS')
    # print("Result:", json.dumps(result, indent=2))