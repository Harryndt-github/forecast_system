"""
==============================================
SCHEDULER (MỚI - PHASE 4)
==============================================
Trách nhiệm:
- Tự động chạy forecast hệ thống theo lịch
- Chạy monitoring/accuracy check sau mỗi forecast run
- Retry logic khi pipeline fail
- Logging run history

Modes:
    1. Single run:   python -m forecast_system.scheduler --once
    2. Daily loop:   python -m forecast_system.scheduler --daily
    3. Custom:       python -m forecast_system.scheduler --interval 6
    
Schedule mặc định:
    - Forecast: 10:00 AM hàng ngày
    - Monitoring: Ngay sau forecast
"""

import sys
import os
import time
import datetime
import argparse
import traceback
import signal

# Ensure import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecast_system.utils.logger import setup_logger
from forecast_system.config.settings import LOG_DIR


class ForecastScheduler:
    """
    Scheduler cho hệ thống forecast.
    
    Responsibilities:
    - Run forecast pipeline theo lịch
    - Run monitoring sau mỗi forecast
    - Track run history
    - Handle errors gracefully
    """
    
    def __init__(self, run_hour: int = 10, run_minute: int = 0, forecast_mode: str = 'daily'):
        self.run_hour = run_hour
        self.run_minute = run_minute
        self.forecast_mode = forecast_mode
        self.logger = setup_logger('scheduler', log_dir=LOG_DIR)
        self.running = True
        self.run_history = []
        
        # Graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        self.logger.info(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
        self.running = False
    
    def run_once(self):
        """
        Chạy 1 lần: Forecast + Monitoring.
        
        Returns:
            bool: True nếu thành công
        """
        start_time = datetime.datetime.now()
        self.logger.info("=" * 60)
        self.logger.info(f"🚀 SCHEDULER: Starting single run at {start_time.strftime('%H:%M:%S')}")
        self.logger.info("=" * 60)
        
        success = False
        
        try:
            # 1. Run forecast
            mode_label = 'FULL (3 tháng)' if self.forecast_mode == 'full' else f'DAILY (30 ngày)'
            self.logger.info(f"\n🔮 PHASE 1: Running Forecast Pipeline [{mode_label}]...")
            from forecast_system.main import main as run_forecast
            run_forecast(forecast_mode=self.forecast_mode)
            
            # 2. Run monitoring
            self.logger.info("\n📊 PHASE 2: Running Monitoring...")
            self._run_monitoring()
            
            success = True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            traceback.print_exc()
        
        # Track run
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        run_entry = {
            'start': start_time.isoformat(),
            'elapsed_seconds': round(elapsed, 1),
            'success': success,
        }
        self.run_history.append(run_entry)
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.logger.info(f"\n{status} | Elapsed: {elapsed/60:.1f} minutes")
        
        return success
    
    def _run_monitoring(self):
        """Chạy monitoring agent."""
        try:
            from forecast_system.agents.monitoring_agent import MonitoringAgent
            from forecast_system.agents.master_file_agent import MasterFileAgent
            from forecast_system.config.settings import MASTER_FILE_NAME
            
            # Load master file
            df = MasterFileAgent.load_or_create(MASTER_FILE_NAME)
            
            if df.empty or len(df) < 10:
                self.logger.warning("Not enough data for monitoring")
                return
            
            # Generate report
            report = MonitoringAgent.generate_full_report(df)
            
            # Print report
            MonitoringAgent.print_report(report, logger_func=self.logger.info)
            
            # Save Excel report
            MonitoringAgent.save_report_excel(df, report)
            
            # Check for critical alerts
            drift = report.get('drift', {})
            if drift.get('has_drift'):
                critical_alerts = [
                    a for a in drift.get('alerts', [])
                    if a.get('level') == 'CRITICAL'
                ]
                if critical_alerts:
                    self.logger.warning(
                        f"🚨 {len(critical_alerts)} CRITICAL alerts detected!"
                    )
            
        except Exception as e:
            self.logger.error(f"Monitoring failed: {e}")
            traceback.print_exc()
    
    def run_daily_loop(self):
        """
        Chạy vòng lặp hàng ngày.
        Forecast chạy lúc self.run_hour:self.run_minute mỗi ngày.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"🕐 SCHEDULER: Daily mode | Run at {self.run_hour:02d}:{self.run_minute:02d}")
        self.logger.info(f"   Press Ctrl+C to stop")
        self.logger.info("=" * 60)
        
        last_run_date = None
        
        while self.running:
            now = datetime.datetime.now()
            today = now.date()
            
            # Check if it's time to run
            should_run = (
                now.hour == self.run_hour and
                now.minute >= self.run_minute and
                last_run_date != today
            )
            
            if should_run:
                self.logger.info(f"\n⏰ Scheduled run triggered at {now.strftime('%H:%M:%S')}")
                
                success = self.run_once()
                last_run_date = today
                
                if not success:
                    # Retry after 30 minutes
                    self.logger.info("⏳ Retrying in 30 minutes...")
                    retry_time = datetime.datetime.now() + datetime.timedelta(minutes=30)
                    
                    while (
                        self.running and 
                        datetime.datetime.now() < retry_time
                    ):
                        time.sleep(10)
                    
                    if self.running:
                        self.logger.info("🔄 Retry attempt...")
                        self.run_once()
            
            # Sleep 30 seconds between checks
            if self.running:
                time.sleep(30)
        
        self.logger.info("\n🏁 Scheduler stopped.")
    
    def run_interval_loop(self, interval_hours: float):
        """
        Chạy theo interval cố định (mỗi N giờ).
        
        Args:
            interval_hours: Khoảng cách giữa các lần chạy (giờ)
        """
        self.logger.info("=" * 60)
        self.logger.info(f"🔄 SCHEDULER: Interval mode | Every {interval_hours}h")
        self.logger.info(f"   Press Ctrl+C to stop")
        self.logger.info("=" * 60)
        
        while self.running:
            self.run_once()
            
            if not self.running:
                break
            
            next_run = datetime.datetime.now() + datetime.timedelta(hours=interval_hours)
            self.logger.info(f"\n⏳ Next run at {next_run.strftime('%H:%M:%S')}")
            
            # Sleep in small increments to allow graceful shutdown
            wait_seconds = int(interval_hours * 3600)
            for _ in range(wait_seconds // 10):
                if not self.running:
                    break
                time.sleep(10)
        
        self.logger.info("\n🏁 Scheduler stopped.")
    
    def print_run_history(self):
        """Print run history."""
        if not self.run_history:
            self.logger.info("No run history yet.")
            return
        
        self.logger.info(f"\n📋 RUN HISTORY ({len(self.run_history)} runs):")
        for i, run in enumerate(self.run_history[-10:], 1):
            status = "✅" if run['success'] else "❌"
            self.logger.info(
                f"   {i}. {status} {run['start']} "
                f"({run['elapsed_seconds']:.0f}s)"
            )


def main():
    """CLI entry point cho scheduler."""
    parser = argparse.ArgumentParser(
        description='AI Forecast System Scheduler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m forecast_system.scheduler --once                     # Daily mode (30 ngày, nhanh)
  python -m forecast_system.scheduler --once --full-forecast     # Full mode (3 tháng, đầy đủ)
  python -m forecast_system.scheduler --daily                    # Daily at 10:00 AM (30 ngày)
  python -m forecast_system.scheduler --daily --full-forecast    # Daily at 10:00 AM (3 tháng)
  python -m forecast_system.scheduler --daily --hour 14          # Daily at 2:00 PM
  python -m forecast_system.scheduler --interval 8               # Every 8 hours
  python -m forecast_system.scheduler --monitor-only             # Only monitoring
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--once', action='store_true',
        help='Run forecast + monitoring once'
    )
    mode_group.add_argument(
        '--daily', action='store_true',
        help='Run daily at specified hour'
    )
    mode_group.add_argument(
        '--interval', type=float, metavar='HOURS',
        help='Run every N hours'
    )
    mode_group.add_argument(
        '--monitor-only', action='store_true',
        help='Only run monitoring (no forecast)'
    )
    
    parser.add_argument(
        '--hour', type=int, default=10,
        help='Hour to run daily (0-23, default: 10)'
    )
    parser.add_argument(
        '--minute', type=int, default=0,
        help='Minute to run daily (0-59, default: 0)'
    )
    parser.add_argument(
        '--full-forecast', action='store_true',
        help='Forecast 3 tháng đầy đủ (mặc định: chỉ 30 ngày)'
    )
    
    args = parser.parse_args()
    forecast_mode = 'full' if args.full_forecast else 'daily'
    
    scheduler = ForecastScheduler(
        run_hour=args.hour,
        run_minute=args.minute,
        forecast_mode=forecast_mode,
    )
    
    if args.once:
        scheduler.run_once()
    elif args.daily:
        scheduler.run_daily_loop()
    elif args.interval:
        scheduler.run_interval_loop(args.interval)
    elif args.monitor_only:
        scheduler._run_monitoring()


if __name__ == '__main__':
    main()
