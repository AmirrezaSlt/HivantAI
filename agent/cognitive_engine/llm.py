from abc import ABC, abstractmethod
from typing import List, Optional

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_response(self, messages: List[dict], max_tokens: int, temperature: float) -> Optional[str]:
        pass
