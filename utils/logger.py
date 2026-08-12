"""
==============================================
STRUCTURED LOGGING
==============================================
Cung cấp logging có cấu trúc cho toàn bộ hệ thống.
- Console output có màu
- File logging với rotation
"""

import logging
import os
import sys
import datetime
from pathlib import Path


class ColorFormatter(logging.Formatter):
    """Custom formatter với emoji + màu cho console"""
    
    COLORS = {
        logging.DEBUG: '\033[36m',      # Cyan
        logging.INFO: '\033[32m',       # Green
        logging.WARNING: '\033[33m',    # Yellow
        logging.ERROR: '\033[31m',      # Red
        logging.CRITICAL: '\033[1;31m', # Bold Red
    }
    RESET = '\033[0m'
    
    ICONS = {
        logging.DEBUG: '🔍',
        logging.INFO: '✅',
        logging.WARNING: '⚠️',
        logging.ERROR: '❌',
        logging.CRITICAL: '🚨',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, '')
        icon = self.ICONS.get(record.levelno, '')
        record.msg = f"{icon} {color}{record.msg}{self.RESET}"
        return super().format(record)


def setup_logger(name='forecast_system', log_dir=None, level=logging.INFO):
    """
    Setup logger với console + file output.
    
    Args:
        name: Logger name
        log_dir: Directory cho log files. None = chỉ console.
        level: Logging level
    
    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Xóa handlers cũ nếu có
    logger.handlers.clear()
    
    # Console Handler (có màu)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = ColorFormatter(
        '%(asctime)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    # File Handler (nếu có log_dir)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        today = datetime.date.today().strftime('%Y-%m-%d')
        file_handler = logging.FileHandler(
            log_path / f"forecast_{today}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logger


def get_logger(name=None):
    """Get existing logger or create child logger"""
    base = 'forecast_system'
    if name:
        return logging.getLogger(f"{base}.{name}")
    return logging.getLogger(base)
