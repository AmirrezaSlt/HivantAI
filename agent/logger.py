import logging
import sys
from typing import Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler

class AgentLogger:
    """Centralized logger for the agent."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.logger = logging.getLogger("agent")
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        log_dir = Path("/var/log/agent")
        log_dir.mkdir(parents=True, exist_ok=True)

        debug_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        info_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Use RotatingFileHandler to rotate logs after they reach 10MB, keeping up to 5 backups.
        debug_handler = RotatingFileHandler(
            str(log_dir / "debug.log"),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(debug_formatter)
        
        info_handler = RotatingFileHandler(
            str(log_dir / "info.log"),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(info_formatter)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(info_formatter)
        
        self.logger.addHandler(debug_handler)
        self.logger.addHandler(info_handler)
        self.logger.addHandler(console_handler)
    
    def setup(self, log_dir: Optional[str] = None):
        """Setup logging directory."""
        log_path = Path(log_dir or "logs")
        log_path.mkdir(parents=True, exist_ok=True)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)

# Global logger instance
logger = AgentLogger()
