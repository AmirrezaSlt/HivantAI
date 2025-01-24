from typing import List
from .config import ToolkitConfig
from .tool import BaseTool

class Toolkit:
    def __init__(self, *args, **kwargs):
        self._config = ToolkitConfig(*args, **kwargs)
        self.tools: List[BaseTool] = self._config.TOOLS
