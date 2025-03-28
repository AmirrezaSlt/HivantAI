from abc import ABC, abstractmethod
from typing import List, Optional, Iterator

class BaseLLMProvider(ABC):
    supports_streaming = False

    def __init__(self):
        pass

    @abstractmethod
    def generate_response(self, messages: List[dict], *args, **kwargs) -> Optional[str]:
        pass

    @abstractmethod
    def stream_response(self, messages: List[dict], *args, **kwargs) -> Iterator[str]:
        pass
