from enum import Enum
from typing import Optional
from pydantic import BaseModel, model_validator

class ResponseType(Enum):
    FINAL = "final"
    TOOL_USE = "tool_use"
    CLARIFICATION = "clarification"

class Response(BaseModel):
    response_type: ResponseType
    content: Optional[str]   = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    question: Optional[str] = None

    @model_validator(mode='after')
    def check_fields(self):
        if self.response_type == ResponseType.FINAL:
            assert self.content is not None, "content is required for FINAL response"
            assert self.tool_name is None and self.tool_input is None and self.question is None
        elif self.response_type == ResponseType.TOOL_USE:
            assert self.tool_name is not None and self.tool_input is not None, "tool_name and tool_input are required for TOOL_USE response"
            assert self.content is None and self.question is None
        elif self.response_type == ResponseType.CLARIFICATION:
            assert self.question is not None, "question is required for CLARIFICATION response"
            assert self.content is None and self.tool_name is None and self.tool_input is None
        return self

class OutputParser:
    @staticmethod
    def parse(response: str) -> Response:
        try:
            return Response.model_validate_json(response)
        except ValueError as e:
            raise ValueError(f"Invalid response structure: {str(e)}")
