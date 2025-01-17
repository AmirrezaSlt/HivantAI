from abc import ABC, abstractmethod
from typing import Dict, Any, Type
import json
from pydantic import BaseModel, ValidationError

class JSONSerializationError(Exception):
    pass

class ToolInfo(BaseModel):
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]

class BaseTool(ABC):
    @property
    @abstractmethod
    def input_model(self) -> Type[BaseModel]:
        """
        Return the Pydantic model class for input validation.
        """
        pass

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
    
    @property
    def info(self) -> ToolInfo:
        return ToolInfo(
            description=self.description,
            inputs=self.input_model.model_json_schema(),
            outputs=self.output_schema
        )

    @abstractmethod
    def _invoke(self, inputs: BaseModel) -> Dict[str, Any]:
        pass

    def invoke(self, **kwargs) -> Dict[str, Any]:
        try:
            validated_inputs = self.input_model(**kwargs)
            outputs = self._invoke(validated_inputs)
            return json.loads(json.dumps(outputs, default=str))
        except ValidationError as e:
            raise ValueError(f"Invalid input: {e}")
        except JSONSerializationError as e:
            raise ValueError(f"Error serializing output: {e}")
