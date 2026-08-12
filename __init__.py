# Forecast System Package

# Windows compatibility: ensure UTF-8 stdout/stderr to avoid UnicodeEncodeError
# when printing Vietnamese characters or emoji on Windows consoles (cp1252)
import sys as _sys
import os as _os
if _os.name == 'nt':
    _os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass
