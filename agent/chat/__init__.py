from abc import ABC, abstractmethod
from typing import List, Optional

class BaseChatProvider(ABC):
    @abstractmethod
    def send_message(self, messages: List[dict], max_tokens: int, temperature: float) -> Optional[str]:
        pass