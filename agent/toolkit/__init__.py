from typing import Dict
from .config import ToolkitConfig
from .tool import BaseTool

class Toolkit:
    def __init__(self, *args, **kwargs):
        self._config = ToolkitConfig(*args, **kwargs)
        self.tools: Dict[str, BaseTool] = {
            tool.id: tool for tool in self._config.TOOLS
        }
