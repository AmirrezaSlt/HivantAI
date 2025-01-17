from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseDataSource(ABC):

    @abstractmethod
    def get_content(self, identifier: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_resource_link(self, uri: str) -> str:
        pass