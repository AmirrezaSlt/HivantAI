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

    @property
    @abstractmethod
    def link(self) -> str:
        """Get document identifier/location"""
        pass

    @property
    @abstractmethod
    def data(self) -> Dict[str, Any]:
        """Get document data content"""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Get document metadata"""
        pass

    def __dict__(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "link": self.link,
            "data": self.data,
            "metadata": self.metadata
        }
    