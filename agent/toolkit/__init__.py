from typing import Dict, Any, Generator
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
        
        # Add code_executor to the tools dictionary if it exists
        if self.code_executor:
            self.tools[self.code_executor.id] = self.code_executor

    def invoke(self, tool_name: str, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronously invoke a tool and return the complete result.
        """
        tool = self.tools.get(tool_name)

        return tool.invoke(**input)
            
    def invoke_stream(self, tool_name: str, input: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Invoke a tool with streaming support, yielding partial results as they become available.
        
        Args:
            tool_name: Name of the tool to invoke
            input: Arguments to pass to the tool
            
        Yields:
            Partial results from the tool as they become available
            
        Raises:
            ValueError: If the tool is not found
        """
        tool = self.tools.get(tool_name)
        if tool.supports_streaming:
            yield from tool.invoke_stream(**input)
        else:
            result = tool.invoke(**input)
            yield result
            
    def supports_streaming(self, tool_name: str) -> bool:
        """
        Check if a specific tool supports streaming.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool supports streaming, False otherwise
        """
        return self.tools[tool_name].supports_streaming
