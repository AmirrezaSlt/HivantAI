from typing import Dict, Any
from .config import ToolkitConfig
from .tool import BaseTool
from .executor import PythonCodeExecutor
import json

class Toolkit:
    def __init__(self, *args, **kwargs):
        self._config = ToolkitConfig(*args, **kwargs)
        self.tools: Dict[str, BaseTool] = {
            tool.id: tool for tool in self._config.TOOLS
        }

        self.code_executor = PythonCodeExecutor(self._config.EXECUTOR) if self._config.EXECUTOR else None

    @property
    def info(self) -> Dict[str, Any]:
        """
        Returns a description of the toolkit's capabilities.
        """
        info = {}
        for tool_name, tool in self.tools.items():
            info[tool_name] = tool.info
        if self.code_executor:
            info["code_executor"] = self.code_executor.info
        return info
    
    def invoke(self, tool_name: str, input: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tools.get(tool_name)
        if tool_name == "code_executor":
            input = self.code_executor.input_model(**input)
            return self.code_executor.execute(**input.model_dump())
        elif tool:
            return tool.invoke(input)
        else:
            raise ValueError(f"Tool {tool_name} not found")
