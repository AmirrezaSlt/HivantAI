from typing import Dict, Any, Optional
from pydantic import BaseModel
import requests
from ...agent.toolkit.tool import BaseTool

class CodeExecutionInput(BaseModel):
    code: str
    timeout: Optional[int] = 5

class CodeExecutionTool(BaseTool):
    def __init__(self, id: str = "code_executor", server_address: str = "http://executor:8000", description: str = "Executes Python code and returns the output or error message") -> None:
        super().__init__(id)
        self.server_address = server_address.rstrip('/')
        self._description = description
        
    @property
    def input_model(self) -> type[BaseModel]:
        return CodeExecutionInput

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "output": {"type": "string"},
                "error": {"type": "string"}
            }
        }

    @property
    def description(self) -> str:
        return self._description

    def _invoke(self, inputs: CodeExecutionInput) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.server_address}/execute",
                json={"code": inputs.code}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Failed to execute code: {str(e)}"}