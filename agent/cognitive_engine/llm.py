from abc import ABC, abstractmethod
from typing import List, Optional, Iterator

class BaseLLMProvider(ABC):
    supports_streaming = False

    def __init__(self):
        pass

    @abstractmethod
    def generate_response(self, messages: List[dict], max_tokens: int, temperature: float) -> Optional[str]:
        pass

    @abstractmethod
    def stream_response(self, messages: List[dict], max_tokens: int, temperature: float) -> Iterator[str]:
        pass
