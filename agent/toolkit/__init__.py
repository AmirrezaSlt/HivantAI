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

    def to_prompt(self) -> str:
        """
        Returns a description of the toolkit's capabilities as a prompt
        to inform the model about available tools and code execution functionality.
        """
        tool_descriptions = []
        
        for tool_id, tool in self.tools.items():
            tool_descriptions.append(f"""
                Tool Name: {tool_id}
                Description: {tool.description}
                Input Schema: {json.dumps(tool.input_schema, indent=2)}
                Output Schema: {json.dumps(tool.output_schema, indent=2)}
            """)

        if self.code_executor:
            tool_descriptions.append(f"""
                Tool Name: code_executor
                Description: {self.code_executor.description}
                Input Schema: {self.code_executor.input_schema}
                Output Schema: {self.code_executor.output_schema}
            """)

        tools_text = "\n\n".join(tool_descriptions) if tool_descriptions else "No tools configured"
        
        return f"""Available Tools: 
        {tools_text}

            To use a tool, provide the tool name and the required inputs according to the input schema.
            Each tool will return outputs matching its output schema.
        """
    
    def invoke_tool(self, tool_name: str, input: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.tools.get(tool_name)
        if tool_name == "code_executor":
            input = self.code_executor.input_model(**input)
            return self.code_executor.execute(**input.model_dump())
        elif tool:
            return tool.invoke(input)
        else:
            raise ValueError(f"Tool {tool_name} not found")
