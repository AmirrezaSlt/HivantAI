from abc import ABC, abstractmethod
from typing import Dict, Any, Type
from pydantic import BaseModel

class JSONSerializationError(Exception):
    pass

class ToolInfo(BaseModel):
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]

class BaseTool(ABC):

    def __init__(self, id: str) -> None:
        self._id = id

    @property
    def id(self) -> str:
        return self._id

    @property
    @abstractmethod
    def input_model(self) -> Type[BaseModel]:
        """
        Return the Pydantic model class for input validation.
        """
        pass

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for the input model.
        """
        return self.input_model.model_json_schema()

    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable dictionary representing the output schema.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def _invoke(self, inputs: BaseModel) -> Dict[str, Any]:
        pass

    def invoke(self, **kwargs) -> Dict[str, Any]:
        return self._invoke(self.input_model(**kwargs))
    
    @property
    def info(self) -> ToolInfo:
        return ToolInfo(
            description=self.description,
            inputs=self.input_schema,
            outputs=self.output_schema
        )

    def __dict__(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema
        }
