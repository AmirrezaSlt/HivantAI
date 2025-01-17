from abc import ABC, abstractmethod
from contextlib import contextmanager

class BaseConnection(ABC):
    @classmethod
    @abstractmethod
    def get_connection(cls, *args, **kwargs):
        """Get or create a connection instance"""
        pass
    
    @abstractmethod
    def close(self):
        """Close the connection and clean up resources"""
        pass
    
    @classmethod
    @contextmanager
    def connection(cls, *args, **kwargs):
        """Context manager for auto-closing connections"""
        conn = cls.get_connection(*args, **kwargs)
        try:
            yield conn
        finally:
            conn.close()