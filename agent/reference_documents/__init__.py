from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseReferenceDocument(ABC):
    def __init__(self, id: str):
        """Initialize resource with an ID."""
        self._id = id

    @property
    def id(self) -> str:
        """Get the document identifier"""
        return self._id

    @abstractmethod
    def get_link(self) -> str:
        """Get document identifier/location"""
        pass

    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """Get document data content"""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get document metadata"""
        pass
