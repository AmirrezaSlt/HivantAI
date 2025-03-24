from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Generator, Union, Optional
from pydantic import BaseModel

class JSONSerializationError(Exception):
    pass

class ToolInfo(BaseModel):
    description: str  # Will include both tool functionality and output format
    inputs: Dict[str, Any]
    supports_streaming: bool = False

class BaseTool(ABC):

    def __init__(self, id: str) -> None:
        self._id = id
        self._supports_streaming = False

    @property
    def id(self) -> str:
        return self._id
        
    @property
    def supports_streaming(self) -> bool:
        """
        Returns whether this tool supports streaming output.
        """
        return self._supports_streaming

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Return a complete description of the tool including:
        - What it does
        - Required and optional inputs
        - Expected output format
        
        Example format:
        '''
        Executes Python code in a sandboxed environment.
        
        Inputs:
        - code: (required) Python code to execute as a string
        - timeout: (optional, default=5) Maximum execution time in seconds
        
        Returns:
        A dictionary containing:
        - output: captured stdout from code execution
        - error: error message if execution failed, null otherwise
        '''
        """
        pass

    @property
    @abstractmethod
    def input_model(self) -> Type[BaseModel]:
        """For input validation only, not for LLM consumption."""
        pass

    @abstractmethod
    def _invoke(self, inputs: BaseModel) -> Any:
        """
        Execute the tool and return the result.
        """
        pass
        
    def _invoke_stream(self, inputs: BaseModel) -> Generator[Dict[str, Any], None, None]:
        """
        Default implementation for tools that don't support streaming.
        Simply wraps the synchronous _invoke method and yields the result once.
        """
        result = self._invoke(inputs)
        yield result

    def invoke(self, **kwargs) -> Any:
        """
        Synchronously invoke the tool and return the complete result.
        """
        return self._invoke(self.input_model(**kwargs))
        
    def invoke_stream(self, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Invoke the tool with streaming, yielding partial results as they become available.
        """
        input_obj = self.input_model(**kwargs)
        yield from self._invoke_stream(input_obj)
    
    @property
    def info(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description
        }

    def __dict__(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "input_schema": self.input_schema,
        }
