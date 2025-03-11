from typing import List, Dict, Any, Optional, Union, Literal, Annotated
from enum import Enum
from pydantic import BaseModel, Field, field_serializer
from agent.input import Input
from agent.toolkit import Toolkit
from agent.retriever.reference_documents import BaseReferenceDocument
import json

class ReasoningStepType(str, Enum):
    INPUT = "input"
    REFERENCE_DOCUMENTS = "reference_documents"
    CLARIFICATION = "clarification"
    TOOL_USE_REQUEST = "tool_use_request"
    TOOL_USE_RESPONSE = "tool_use_response"
    RESPONSE = "response"

class ToolUseRequestStep(BaseModel):
    type: Literal[ReasoningStepType.TOOL_USE_REQUEST] = ReasoningStepType.TOOL_USE_REQUEST
    tool_name: str = Field(description="Name of the tool to use")
    input_data: Dict[str, Any] = Field(description="Input data for the tool")

class ToolUseResponseStep(BaseModel):
    type: Literal[ReasoningStepType.TOOL_USE_RESPONSE] = ReasoningStepType.TOOL_USE_RESPONSE
    output_data: Dict[str, Any] = Field(description="Output data from the tool")

class ClarificationStep(BaseModel):
    type: Literal[ReasoningStepType.CLARIFICATION] = ReasoningStepType.CLARIFICATION
    clarification: str = Field(description="Clarification to the user")

class ResponseStep(BaseModel):
    type: Literal[ReasoningStepType.RESPONSE] = ReasoningStepType.RESPONSE
    response: str = Field(description="Response to the user")

class InputStep(BaseModel):
    type: Literal[ReasoningStepType.INPUT] = ReasoningStepType.INPUT
    input: Input = Field(description="Input of the user")

class RelevantDocumentsStep(BaseModel):
    type: Literal[ReasoningStepType.REFERENCE_DOCUMENTS] = ReasoningStepType.REFERENCE_DOCUMENTS
    reference_documents: List[BaseReferenceDocument] = Field(default_factory=list)

    model_config = {
        "arbitrary_types_allowed": True
    }

    @field_serializer('reference_documents')
    def serialize_documents(self, docs: List[BaseReferenceDocument]) -> List[Dict[str, Any]]:
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
            } for doc in docs
        ]

class LLMResponse(BaseModel):
    step: Annotated[Union[ToolUseRequestStep, ResponseStep, ClarificationStep], Field(discriminator="type")]
    explanation: Optional[str] = Field(default=None, description="Short explanation of why this step was chosen")

class ReasoningState:
    
    def __init__(self, toolkit: Toolkit, system_prompt: str):
        self.toolkit = toolkit
        self.trail = []
        self._system_prompt = system_prompt

    def update_state_from_llm_response(self, response: LLMResponse) -> None:
        """Update the reasoning state from a raw LLM response string."""
        self.trail.append(response)

    def add_tool_use_response(self, output_data: Dict[str, Any]) -> None:
        self.trail.append(ToolUseResponseStep(
            output_data=output_data
        ))

    def update_state_from_input(self, input: Input, reference_documents: List[BaseReferenceDocument]) -> None:
        self.trail.append(InputStep(
            input=input
        ))
        if reference_documents:
            self.trail.append(RelevantDocumentsStep(
                reference_documents=reference_documents
            ))
    
    @property
    def system_prompt(self) -> str:
        prompt = [
            self._system_prompt,
            "",
            "Format your response using the following schema:",
            json.dumps(LLMResponse.model_json_schema(), indent=2),
            ""
        ]
        
        if self.toolkit:
            prompt.append(self.toolkit.to_prompt())
        return "\n".join(prompt)
    
    @property
    def messages(self) -> List[Dict[str, str]]:
        """Returns the message list for the LLM with the current context"""
        messages = [{"role": "system", "content": self.system_prompt}]
        for step in self.trail:
            content = step.model_dump_json()
            if isinstance(step, InputStep):
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})
        return messages
